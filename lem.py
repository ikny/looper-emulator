from typing import Any
import numpy.typing as npt

import sounddevice as sd
import numpy as np
import soundfile as sf
import threading


SAMPLERATE = 44100  # [samples per second]
BLOCKSIZE = 10000   # [samples]
CHANNELS = 2
LATENCY = 0

DTYPE = np.int16
STR_DTYPE = "int16"

METRONOME_SAMPLE_PATH = "lib/samples/metronome.wav"


class Lem():
    """Handles the logic of the looper emulator. 
    While LoopStreamManager only knows how to record a track and update its tracks data, 
    Lem operates on a higher level, featuring a metronome,
    adding, removing and modifying individual tracks. 
    """

    def __init__(self) -> None:
        """Initialize a new instance of Lem (looper emulator).
        """
        self._stream_manager: LoopStreamManager = LoopStreamManager()

        self._tracks: list[npt.NDArray[DTYPE]] = []

        self._metronome_volume = 1/8

    def initialize_stream(self, bpm: int) -> None:
        """Prepare the metronome, so the user can start to record tracks, and start a stream.

        Args:
            bpm (int): The number of metronome beats per minute.
        """
        # initialize the metronome sample and start a stream
        metronome = self.metronome_generator(
            bpm=bpm, path=METRONOME_SAMPLE_PATH)
        self._tracks.append(metronome)
        self._update_tracks()

        self._stream_manager.start_stream()

    def terminate(self) -> None:
        """Delegate the closing of the stream to stream manager.
        """
        self._stream_manager.end_stream()

    def metronome_generator(self, bpm: int, path: str) -> npt.NDArray[DTYPE]:
        """Cut or fill the metronome sample so that it is long exactly one beat of the given BPM.

        Args:
            bpm (int): The desired tempo.
            path (str): Path to the metronome sample.

        Returns:
            npt.NDArray[DTYPE]: The resulting metronome sample long exactly one beat of the given BPM.
        """
        sample: npt.NDArray[DTYPE]
        # TODO: handle errors when getting the sample
        sample, samplerate = sf.read(file=path, dtype=STR_DTYPE)

        desired_len = int((60*samplerate)/bpm)
        self.len_beat = desired_len

        if len(sample) <= desired_len:
            # rounding desired_len introduces a slight distortion of bpm
            sample = np.concatenate(
                (sample, np.zeros(shape=(desired_len-len(sample), CHANNELS), dtype=DTYPE)))
        else:
            sample = sample[:desired_len]

        # adjust volume
        sample = (sample*self._metronome_volume).astype(dtype=DTYPE)

        return sample

    def start_recording(self) -> None:
        """Delegate the start to its stream manager.
        """
        self._stream_manager.start_recording()

    def stop_recording(self) -> None:
        """Delegate the stop to its stream manager. 
        This returns the new track, which is then passed to post production function.
        """
        recorded_track = self._stream_manager.stop_recording()
        self.post_production(recorded_track=recorded_track)

    def post_production(self, recorded_track: npt.NDArray[DTYPE]) -> None:
        """Cut/fill newly recorded track to whole bpm and add it to tracks.

        Args:
            recorded_track (npt.NDArray[DTYPE]): The audio data to be modified.
        """
        remainder = len(recorded_track) % self.len_beat

        if remainder > self.len_beat/2:
            zeros = np.zeros(
                shape=(self.len_beat-remainder, CHANNELS), dtype=DTYPE)
            recorded_track = np.concatenate([recorded_track, zeros])
        elif remainder <= self.len_beat/2:
            recorded_track = recorded_track[:len(
                recorded_track)-remainder]

        if len(recorded_track):
            self._tracks.append(recorded_track)
            self._update_tracks()

    def delete_track(self, idx: int) -> None:
        """Removes the track with index idx.

        Args:
            idx (int): The index of the track which is being deleted.
        """
        self._tracks.pop(idx+1) # +1 because of the metronome track
        self._update_tracks()

    def _update_tracks(self) -> None:
        """Pass self._tracks to update_tracks function of its stream manager.
        """
        self._stream_manager.update_tracks(tracks=self._tracks)


class LoopStreamManager():
    """Manages the sounddevice stream thread including error handling.
    Plays the content of _tracks in a loop. 
    Can record a new track and update _tracks in a thread-safe way.

    For more information about sounddevice library see sounddevice documentation:
    https://python-sounddevice.readthedocs.io/en/0.4.6/api/index.html.
    """

    def __init__(self) -> None:
        """Initialize a new LoopStreamManager object.
        """
        # flags
        self._recording = False
        self._stream_active = True
        self._current_frame = 0

        self._stream_thread: threading.Thread
        # the audio data to be played
        self._tracks: list[npt.NDArray[DTYPE]] = []
        # the synchronized backup of _tracks which is used as a data source when the _tracks are being updated
        self._tracks_copy: list[npt.NDArray[DTYPE]]
        self._tracks_lock: threading.Lock = threading.Lock()

        self._recorded_track: npt.NDArray[DTYPE] = np.empty(
            shape=(0, CHANNELS), dtype=DTYPE)

    def start_stream(self) -> None:
        """Make a new thread, in which the stream will be active
        """
        self._stream_thread = threading.Thread(target=self.main)
        self._stream_thread.start()

    def end_stream(self) -> None:
        """Set stream_active to false. If the stream was already initialized, wait for the thread to end. 
        """
        self._stream_active = False
        try:
            self._stream_thread.join()
        except AttributeError:
            return

    def start_recording(self) -> None:
        """Set the _recording flag to True. This works because 
        the callback function checks the flag when deciding whether to store indata.
        """
        self._recording = True

    def stop_recording(self) -> npt.NDArray[DTYPE]:
        """Set the _recording flag to False and prepare _recording_track for new recording.

        Returns:
            npt.NDArray[DTYPE]: The recorded track.
        """
        self._recording = False
        recorded_track = self._recorded_track
        self._recorded_track = np.empty(shape=(0, CHANNELS), dtype=DTYPE)
        return recorded_track

    def update_tracks(self, tracks: list[npt.NDArray[DTYPE]]) -> None:
        """Update its data in a thread safe manner using lock. First update the backup, 
        from which will the callback read while _tracks are being updated.

        Args:
            tracks (npt.NDArray[DTYPE]): New data by which _tracks will be replaced.
        """
        self._tracks_copy = self._tracks
        with self._tracks_lock:
            self._tracks = tracks

    def main(self) -> None:
        """Open a sounddevice stream and keep it active while flag _stream_active is true.
        """

        def callback(indata: npt.NDArray[DTYPE], outdata: npt.NDArray[DTYPE],
                     frames: int, time: Any, status: sd.CallbackFlags) -> None:
            """The callback function, which is called by the stream every time it needs audio data.
            As mentioned in the sounddevice documentation, the callback has to have these arguments and return None.
            For more information about callback see: https://python-sounddevice.readthedocs.io/en/0.4.6/api/streams.html#streams-using-numpy-arrays.

            Args:
                indata (npt.NDArray[DTYPE]): The input buffer. A two-dimensional numpy array with a shape of (frames, channels).
                outdata (npt.NDArray[DTYPE]): The output buffer. A two-dimensional numpy array with a shape of (frames, channels).
                frames (int): The number of frames to be processed (same as the length of input and output buffers).
                time (Any): A timestamp of the capture of the first indata frame.
                status (sd.CallbackFlags): CallbackFlags object, indicating whether input/output under/overflow is happening.
            """

            # TODO: replace this with error handling
            if status:
                print(status)

            if not self._tracks_lock.locked():
                tracks = self._tracks
            else:
                tracks = self._tracks_copy

            if self._recording:
                self._recorded_track = np.concatenate(
                    [self._recorded_track, indata])

            # TODO: implement a "numpy circular buffer"? - ease up slicing, but not at cost of speed... 
            # also bonus points for IT datastructure lol
            sliced_data = [indata]
            # slice
            for track in tracks:
                start = self._current_frame % len(track)
                end = (self._current_frame+frames) % len(track)
                if end < start:
                    track_slice = np.concatenate(
                        (track[start:], track[:end]))
                else:
                    track_slice = track[start:end]
                sliced_data.append(track_slice)

            mixed_data = np.mean(a=sliced_data, axis=0)

            outdata[:] = mixed_data

            # TODO: if necessary, keep the current_frame small enough not to make problems
            self._current_frame += frames

        with sd.Stream(samplerate=SAMPLERATE, blocksize=BLOCKSIZE, dtype=STR_DTYPE,
                       channels=CHANNELS, callback=callback):
            # TODO: is there another way how to keep the stream alive?
            while self._stream_active:
                pass
