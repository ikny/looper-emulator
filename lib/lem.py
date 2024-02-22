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
from constants import *
from tracks import RecordedTrack, PlayingTrack
from custom_exceptions import IncompleteRecordedTrackError, InvalidSamplerateError
from utils import Queue, AudioCircularBuffer, UserRecordingEvents, on_beat, is_in_first_half_of_beat


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
            ZeroDivisionError: If bpm = 0.
        """
        sample: npt.NDArray[DTYPE]
        sample, samplerate = sf.read(
            file=METRONOME_SAMPLE_PATH, dtype=STR_DTYPE)
        if samplerate != SAMPLERATE:
            raise InvalidSamplerateError()

        if len(sample) <= self._len_beat:
            sample = np.concatenate(
                (sample, np.zeros(shape=(self._len_beat-len(sample), CHANNELS), dtype=DTYPE)))
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
        This returns the new track, which is then passed to post production method.

        Returns:
            bool: Passes the value returned by the post_production method.
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
        """
        if len_beat < 0:
            raise ValueError("len_beat must not be negative")
        self._len_beat = len_beat
        self._current_frame = 0

        self._stream_active = True

        self._recording = False
        self._stopping_recording = False
        self._event_queue = Queue()

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
            tracks (npt.NDArray[DTYPE]): New data by which _tracks will be replaced.
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
        # TODO: check all the documentation for correct types and info
        """Sets the _recording flag to False. 
        Waits until the recording stops and afterwards prepares _recording_track for new recording.

        Returns:
            npt.NDArray[DTYPE]: The recorded track.
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
        """Cut/fill newly recorded track to whole bpm and add it to tracks.

        Args:
            recorded_track (npt.NDArray[DTYPE]): The audio data to be modified.

        Returns:
            bool: True if the track was long at least one beat, thus it was actually added into tracks. 
            False if the rounding resulted in an empty track.
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

        if start - first < half_beat:
            start = 0
        else:
            start = self._len_beat
            first += self._len_beat

        if (stop - first) % self._len_beat < half_beat:
            stop = length - self._len_beat
        else:
            stop = length

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
        """
        def slice_and_mix(indata: npt.NDArray[DTYPE], frames: int) -> npt.NDArray[DTYPE]:
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
            As mentioned in the sounddevice documentation, the callback has to have these arguments and return None.
            For more information about callback see: https://python-sounddevice.readthedocs.io/en/0.4.6/api/streams.html#streams-using-numpy-arrays.

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
        # finish the recording and prepare for new one
        logger.debug("Finishing the recording.")
        self._recorded_tracks_queue.push(item=self._recorded_track)
        self._prepare_new_recording()

    def _prepare_new_recording(self) -> None:
        logger.debug("Preparing for new recording.")
        self._recorded_track = RecordedTrack()
        self._recording = False
        self._stopping_recording = False
