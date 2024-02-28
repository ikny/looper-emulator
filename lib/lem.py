# types/dev
from typing import Any, Optional
import numpy.typing as npt
import logging
# libs
import sounddevice as sd
import numpy as np
import soundfile as sf
import threading
from time import sleep
# parts of project
from .constants import *
from .tracks import RecordedTrack, PlayingTrack
from .custom_exceptions import IncompleteRecordedTrackError, InvalidSamplerateError
from .utils import Queue, AudioCircularBuffer, UserRecordingEvents, on_beat, is_in_first_half_of_beat


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(levelname)s: %(asctime)s %(name)s: %(message)s", level=logging.DEBUG)


class Lem():
    """Handles the logic of the looper emulator. 
    While LoopStreamManager only knows how to record a track and update its tracks data, 
    Lem operates on a higher level, featuring a metronome,
    adding, removing and modifying individual tracks. 
    """

    def __init__(self, bpm: int) -> None:
        """Initialize a new instance of Lem (looper emulator).

        Args:
            bpm (int): Beats per minute. This class presumes that 0 < bpm < musically reasonable value (400).

        Raises:
            ZeroDivisionError: If bpm = 0.
        """
        # rounding to int introduces a slight distortion of bpm
        self._len_beat = int(60*SAMPLERATE/bpm)
        self._stream_manager = LoopStreamManager(len_beat=self._len_beat)
        self._tracks: list[PlayingTrack] = []

        self.initialize_metronome()
        self.start_stream()

    def initialize_metronome(self) -> None:
        """Prepare the metronome so that the sample is long exactly one beat of the set BPM.

        Raises:
            InvalidSamplerateError: If the samplerate of the audio file on path is not the same as SAMPLERATE.
            soundfile.LibsndfileError: If the file on path could not be opened.
            TypeError: If the dtype could not be recognized.
        """
        sample: npt.NDArray[DTYPE]
        sample, samplerate = sf.read(
            file=METRONOME_SAMPLE_PATH, dtype=STR_DTYPE)
        if samplerate != SAMPLERATE:
            raise InvalidSamplerateError()

        if len(sample) <= self._len_beat:
            sample = np.concatenate(
                (sample, np.zeros(shape=(self._len_beat-len(sample), CHANNELS), dtype=DTYPE))) # type: ignore
        else:
            sample = sample[:self._len_beat]

        self._tracks.append(PlayingTrack(data=sample))
        self._update_tracks()

    def start_stream(self) -> None:
        """Delegate the start to the stream manager.
        """
        self._stream_manager.start_stream()

    def terminate(self) -> None:
        """Delegate the closing of the stream to stream manager.
        """
        self._stream_manager.end_stream()

    def start_recording(self) -> None:
        """Delegate the start to its stream manager.
        """
        self._stream_manager.start_recording()

    def stop_recording(self) -> bool:
        """Delegate the stop to its stream manager. 
        If the recorded track is long at least one beat, it is returned an appended into tracks.

        Returns:
            bool: True if the track was long at least one beat, thus it was appended into tracks. 
            False if rounding to whole beats resulted in zero length track.
        """
        recorded_track = self._stream_manager.stop_recording()
        if not recorded_track:
            return False
        logger.debug("We have a new track! Updating the tracks...")
        self._tracks.append(recorded_track)
        self._update_tracks()
        return True

    def delete_track(self, idx: int) -> None:
        """Removes the track on index idx+1, because the first track is the metronome sample.

        Args:
            idx (int): The index of the track which is being deleted.
        """
        self._tracks.pop(idx+1)
        self._update_tracks()

    def _update_tracks(self) -> None:
        """Pass self._tracks to update_tracks method of its stream manager.
        """
        self._stream_manager.update_tracks(tracks=self._tracks)


class LoopStreamManager():
    """Manages the sounddevice stream thread including error handling.
    Plays the content of _tracks in a loop. 
    Can record a new track and update _tracks in a thread-safe way.

    For more information about sounddevice library see sounddevice documentation:
    https://python-sounddevice.readthedocs.io/en/0.4.6/api/index.html.
    """

    def __init__(self, len_beat: int) -> None:
        """Initialize a new LoopStreamManager object.

        Args:
            len_beat (int): Length of one beat in frames.

        Raises:
            ValueError: If len_beat < 0.
        """
        if len_beat < 0:
            raise ValueError("len_beat must not be negative")
        self._len_beat = len_beat
        self._current_frame = 0
        self._stream_active = True

        self._event_queue = Queue()
        self._recording = False
        self._stopping_recording = False

        # audio data
        self._stream_thread: Optional[threading.Thread] = None
        self._tracks: list[PlayingTrack] = []
        self._tracks_copy: list[PlayingTrack] = self._tracks
        self._tracks_lock: threading.Lock = threading.Lock()

        self._last_beat = AudioCircularBuffer(
            length=self._len_beat, channels=CHANNELS, dtype=DTYPE)
        self._recorded_track = RecordedTrack()
        self._recorded_tracks_queue = Queue()

    def start_stream(self) -> None:
        """Make a new thread, in which the stream will be active
        """
        self._stream_thread = threading.Thread(target=self.main)
        self._stream_thread.start()

    def end_stream(self) -> None:
        """Set stream_active to false and wait for the stream thread to end. 
        If the stream was not initialized yet, finishes without further action.
        """
        self._stream_active = False
        if self._stream_thread:
            self._stream_thread.join()

    def update_tracks(self, tracks: list[PlayingTrack]) -> None:
        """Update its data in a thread safe manner using lock. First update the backup, 
        from which will the callback read while _tracks are being updated.

        Args:
            tracks (list[PlayingTrack]): New data by which _tracks will be replaced.
        """
        self._tracks_copy = self._tracks
        with self._tracks_lock:
            self._tracks = tracks

    def start_recording(self) -> None:
        """Set the _recording flag to True. This works because 
        the callback method checks the flag when deciding whether to store indata.
        """
        logger.debug("START event pushed to queue.")
        self._event_queue.push(UserRecordingEvents.START)

    def stop_recording(self) -> Optional[PlayingTrack]:
        """Push the stop event into the event queue, then wait until the recording stops 
        and pass the track to the post_production method.

        Returns:
            Optional[PlayingTrack]: If the rounding to whole BPM in the post_production method 
            resulted in an empty track, return None. Otherwise return instance of PlayingTrack.
        """
        self._event_queue.push(UserRecordingEvents.STOP)
        logger.debug("STOP event pushed to queue.")

        while True:
            if not self._recorded_tracks_queue.empty():
                break
            sleep(0.001)
        raw_track: RecordedTrack = self._recorded_tracks_queue.pop()
        logger.debug(
            "Passing the recorded track to the post_production team...")
        recorded_track = self.post_production(raw_track)
        return recorded_track

    def post_production(self, recorded_track: RecordedTrack) -> Optional[PlayingTrack]:
        """This method gets a recorded track including the whole beats within which the START and STOP events happened.
        Furthermore, the recorded track carries information about the time (in frames) of the first recorded frame 
        and times of the START and STOP events.

        It rounds the times of the events to the nearest whole beat (e.g. if the user started the recording late - in the first 
        half of a beat - this method ensures that the recording "was actually started" precisely on the beat the user missed).
        The track is then cut according to these rounded times, resulting in tolerance for both user and program caused time imperfections.

        Args:
            recorded_track (RecordedTrack): The audio data to be modified.

        Raises:
            IncompleteRecordedTrackError: If one of the recorded track attributes was not set.

        Returns:
            Optional[PlayingTrack]: If the rounding to whole BPM resulted in an empty track, return None. 
            Otherwise return instance of PlayingTrack.
        """
        if not recorded_track.is_complete():
            logger.error(f"""Recorded track is not complete: 
                         first is {recorded_track.first_frame_time},
                         start is {recorded_track.start_rec_time},
                         stop is {recorded_track.stop_rec_time},
                         data is {recorded_track.data}.""")
            raise IncompleteRecordedTrackError(
                "RecordedTrack must be complete to be modified in post_production! See RecordedTrack.is_complete().")
        logger.debug(
            f"The frames over are: {len(recorded_track.data)%self._len_beat}. This should be zero!")

        first: int = recorded_track.first_frame_time  # type: ignore
        start: int = recorded_track.start_rec_time  # type: ignore
        stop: int = recorded_track.stop_rec_time  # type: ignore
        data = recorded_track.data
        length = len(data)
        half_beat = int(self._len_beat/2)
        # round the start
        if start - first < half_beat:
            start = 0
        else:
            start = self._len_beat
            first += self._len_beat
        # round the stop
        if (stop - first) % self._len_beat < half_beat:
            stop = length - self._len_beat
        else:
            stop = length
        # cut the audio
        data = data[start:stop]

        if len(data):
            logger.debug(
                f"The track was rounded successfully. It is long {len(data)/self._len_beat} beats.")
            return PlayingTrack(
                data=data, playing_from_frame=first)
        logger.debug("The resulting track had length zero.")
        return None

    """ The following methods are used in a separate thread. """

    def main(self) -> None:
        """Open a sounddevice stream and keep it active while flag _stream_active is true.
        The stream uses default input and output devices.
        """
        def slice_and_mix(indata: npt.NDArray[DTYPE], frames: int) -> npt.NDArray[DTYPE]:
            """Takes indata and an appropriate slice of data from every PlayingTrack in tracks. 
            These are then mixed together using arithmetic mean (at every frame the values of different tracks are averaged).

            Args:
                indata (npt.NDArray[DTYPE]): The audio input, which is mixed to the other tracks.
                frames (int): The length of the slices to be made (which equals to the length of indata).

            Returns:
                npt.NDArray[DTYPE]: All signals mixed in one track (slice) long frames.
            """
            if not self._tracks_lock.locked():
                tracks = self._tracks
            else:
                tracks = self._tracks_copy

            sliced_data = [indata]
            for track in tracks:
                sliced_data.append(track.slice(
                    from_frame=self._current_frame, frames=frames))

            mixed_data: npt.NDArray[DTYPE] = np.mean(a=sliced_data, axis=0)
            return mixed_data

        def callback(indata: npt.NDArray[DTYPE], outdata: npt.NDArray[DTYPE],
                     frames: int, time: Any, status: sd.CallbackFlags) -> None:
            """The callback method, which is called by the stream every time it needs audio data. 
            This function contains the core logic of the audio manipulation and user event handling.

            Firstly, it checks for output underflow - a state where the outdata are not supplied quickly enough.
            In case of output underflow the output is dropped (filled with zeros) in order to compensate for the underflow.
            However, this delays the handling of user events and could result in an unpleasant recording/listening experience.

            Secondly, the user events are handled. Theoratically, only one event can be handled in one callback call,
            however in normal usage of the looper emulator the events are not expected to come nowhere near this capacity.
            On a START recording event the data from the last beat buffer are taken as a base for the recorded track, 
            and the recording flag is raised in order to store (record) the indata in further calls to callback 
            (until the recording is finished). If another start event comes before an old recording could be finished,
            the old recording is discarded and replaced by the new one.
            If a STOP recording event comes in the first half of a beat, the remaining frames of the last beat are filled 
            with zeros and the recording is finished. If the event comes in the second half of a beat, the rest of this beat
            is recorded and the recording is finished afterwards.

            As mentioned in the sounddevice documentation, the callback has to have these arguments and return None.
            For more information about callbacks and stream see: 
            https://python-sounddevice.readthedocs.io/en/0.4.6/api/streams.html#streams-using-numpy-arrays.

            Args:
                indata (npt.NDArray[DTYPE]): The input buffer. A two-dimensional numpy array with a shape of (frames, channels).
                outdata (npt.NDArray[DTYPE]): The output buffer. A two-dimensional numpy array with a shape of (frames, channels).
                frames (int): The number of frames to be processed (same as the length of input and output buffers).
                time (Any): A timestamp of the capture of the first indata frame.
                status (sd.CallbackFlags): CallbackFlags object, indicating whether input/output under/overflow is happening.
            """
            # handle errs
            if status.output_underflow:
                outdata.fill(0)
                return

            # These events change the state
            if not self._event_queue.empty():
                event = self._event_queue.pop()
                if event == UserRecordingEvents.START:
                    logger.debug("User started the recording.")
                    self._initialize_recording()
                elif event == UserRecordingEvents.STOP:
                    logger.debug("User stopped the recording.")
                    self._stop_recording()

            if self._recording:
                self._recorded_track.append(data=indata)
            if self._stopping_recording and on_beat(current_frame=self._current_frame, len_beat=self._len_beat, frames=frames):
                # cut off the few frames over a beat
                self._recorded_track.data = self._recorded_track.data[:len(
                    self._recorded_track.data)-len(self._recorded_track.data) % self._len_beat]
                self._finish_recording()

            # this happens every callback
            self._last_beat.write(data=indata)
            outdata[:] = slice_and_mix(indata=indata, frames=frames)
            self._current_frame += frames

        with sd.Stream(samplerate=SAMPLERATE, blocksize=BLOCKSIZE, dtype=STR_DTYPE, channels=CHANNELS, callback=callback):
            while self._stream_active:
                sleep(1)

    def _initialize_recording(self) -> None:
        """Initialize the recorded track with the data from the last beat buffer. 
        Moreover, note the times of the first frame of the recorded track and the START recording event.
        If the old recording was not finished yet, discard it and initialize a new one.
        """
        logger.debug("Initializing recording.")
        if self._stopping_recording:
            # overwrite the old one
            logger.debug(
                "Still recording: overwriting the unfinished recording.")
            self._prepare_new_recording()
        # initialize recording
        self._recorded_track.first_frame_time = self._current_frame-self._last_beat.position()
        self._recorded_track.start_rec_time = self._current_frame
        self._recorded_track.append(data=self._last_beat.start_to_index())
        self._recording = True

    def _stop_recording(self) -> None:
        """ If the current_frame is in the first half of a beat, fill the remaining frames of the last beat of the recorded track 
        with zeros and finish the recording. If the current_frame is in the second half of a beat, set _stopping_recording 
        flag to True. The callback will finish the recording on the next beat.
        """
        logger.debug("Stopping the recording.")
        # note when the stop_recording came
        self._recorded_track.stop_rec_time = self._current_frame

        if is_in_first_half_of_beat(current_frame=self._current_frame, len_beat=self._len_beat):
            logger.debug(
                "The user's stop came in the first half of beat, finishing immediately.")
            # fill the track to a beat
            self._recorded_track.append(np.zeros(shape=(
                self._len_beat-self._current_frame % self._len_beat, CHANNELS), dtype=DTYPE))
            self._finish_recording()
        else:
            logger.debug(
                "The user's stop came in the second half of a beat, waiting until the end.")
            self._stopping_recording = True

    def _finish_recording(self) -> None:
        """Finish the recording and prepare space for a new one.
        """
        logger.debug("Finishing the recording.")
        self._recorded_tracks_queue.push(item=self._recorded_track)
        self._prepare_new_recording()

    def _prepare_new_recording(self) -> None:
        """Clean up after the old recording in order for a new one to be started.
        """
        logger.debug("Preparing for new recording.")
        self._recorded_track = RecordedTrack()
        self._recording = False
        self._stopping_recording = False
