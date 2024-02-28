# types/dev
from typing import Optional
import numpy.typing as npt
import logging
# libs
import numpy as np
# other parts of this project
from .constants import *


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(levelname)s: %(asctime)s %(name)s: %(message)s", level=logging.DEBUG)


class Track():
    """A base class for other types of track.
    """

    def __init__(self, data: npt.NDArray[DTYPE]) -> None:
        """Initialize a new instance of Track.

        Args:
            data (npt.NDArray[DTYPE], optional): The audio data. Defaults to np.empty((0, CHANNELS), dtype=DTYPE).
        """
        self.data = data


class RecordedTrack(Track):
    """A naive track which is currently being recorded. 
    Does not check whether the values it holds make sense (stop_rec_time can be before start_rec_time etc.). 
    """

    def __init__(self, data: npt.NDArray[DTYPE] = np.empty((0, CHANNELS), dtype=DTYPE)) -> None:
        """Initialize an instance of recorded track.

        Args:
            data (npt.NDArray[DTYPE], optional): The audio data. Defaults to np.empty((0, CHANNELS), dtype=DTYPE).
        """
        super().__init__(data)
        self.first_frame_time: Optional[int] = None
        self.start_rec_time: Optional[int] = None
        self.stop_rec_time: Optional[int] = None

    def append(self, data: npt.NDArray[DTYPE]) -> None:
        """Append the data to data property.

        Args:
            data (npt.NDArray[DTYPE]): The audio data to be appended.
        """
        self.data = np.concatenate([self.data, data]) # type: ignore

    def is_complete(self) -> bool:
        """Checks whether all the properties have been set and length of the data is not zero as when initialized.

        Returns:
            bool: True if all the properties have been set and length of the data is not zero. False otherwise.
        """
        return \
            self.first_frame_time is not None \
            and self.start_rec_time is not None \
            and self.stop_rec_time is not None \
            and self.data.size != 0


class PlayingTrack(Track):
    """A track which is beaing played. Has audio data, knows when it started playing 
    (which is necessary to know which part of track should be played at any point in time).
    """

    def __init__(self, data: npt.NDArray[DTYPE], playing_from_frame: int = 0) -> None:
        """Initialize an instance of PlayingTrack.

        Args:
            data (npt.NDArray[DTYPE]): The audio data.
            playing_from_frame (int, optional): The frame from which the track is playing. Defaults to 0.
        """
        super().__init__(data)
        self._playing_from_frame = playing_from_frame
        self._length = len(data)

    def set_playing_from_frame(self, playing_from_frame: int) -> None:
        """Sets the frame from which the track is playing.

        Args:
            playing_from_frame (int): The frame from which the track should play.
        """
        self._playing_from_frame = playing_from_frame

    def slice(self, from_frame: int, frames: int) -> npt.NDArray[DTYPE]:
        """Returns a slice of self.data long frames. 
        The position of the slice is determined by from_frame, supposing that the track plays in a loop.

        Args:
            from_frame (int): The first frame of the slice.
            frames (int): The length of the returned slice. 

        Returns:
            npt.NDArray[DTYPE]: The slice of audio data to be played
        """
        from_frame -= self._playing_from_frame
        start = from_frame % self._length
        end = (from_frame+frames) % self._length

        track_slice: npt.NDArray[DTYPE]
        if end < start:
            track_slice = np.concatenate(
                (self.data[start:], self.data[:end])) # type: ignore
        else:
            track_slice = self.data[start:end]
        return track_slice
