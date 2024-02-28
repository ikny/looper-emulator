# types
from typing import Any, Callable
import numpy.typing as npt
import numpy as np
# libs
from enum import Enum
from logging import Logger


class Queue():
    """A simple queue that can only push and pop.
    """

    def __init__(self) -> None:
        """Initialize a new instance of Queue.
        """
        self._items: list[Any] = []

    def push(self, item: Any) -> None:
        """Push item to the end of a queue.

        Args:
            item (Any): The item to be pushed.
        """
        self._items.append(item)

    def pop(self) -> Any:
        """Pop the first item from a queue and return it.

        Returns:
            Any: The returned item.
        """
        return self._items.pop(0)

    def empty(self) -> bool:
        """Check whether the Queue is empty.

        Returns:
            bool: True if the Queue is empty, False otherwise.
        """
        return not bool(self._items)


class AudioCircularBuffer():
    """A simple circular buffer. Operates on 2D numpy array with the specified dtype and size.
    Fills the data with write() and on overflow starts to rewrite the data from the beginning.
    This buffer's primary purpose is to be used in audio, for example a stereo buffer always 
    containing the last second of a signal etc.
    """

    def __init__(self, length: int, channels: int, dtype: npt.DTypeLike) -> None:
        """Initialize a new instance of AudioCircularBuffer

        Args:
            length (int): _description_
            channels (int): _description_
            dtype (npt.DTypeLike): _description_
        """
        self._length = length
        self._data = np.empty(
            shape=(self._length, channels), dtype=dtype)
        self._index = 0

    def write(self, data: npt.NDArray) -> None:  # type: ignore
        """Write the data into the buffer. In one call it cannot write more data that its length.

        Args:
            data (npt.NDArray): The data to be written.

        Raises:
            NotImplementedError: If length of data > set length of itself.
        """
        len_data = len(data)
        total_len = self._index + len_data
        if len_data > self._length:
            raise NotImplementedError(
                "Data too long! This circular buffer cannot write longer data than its set length.")
        if total_len > self._length:
            left = self._length - self._index
            overflown = len_data - left
            self._data[self._index:] = data[:left]
            self._data[:overflown] = data[left:]
        else:
            self._data[self._index:total_len] = data
        self._index = total_len % self._length

    def start_to_index(self) -> npt.NDArray:  # type: ignore
        """Returns the data from start to current index.

        Returns:
            npt.NDArray: The returned data.
        """
        return self._data[:self._index] # type: ignore

    def position(self) -> int:
        """Returns the current position of _index.

        Returns:
            int: Current index.
        """
        return self._index


class UserRecordingEvents(Enum):
    """The user of Lem can only start or stop recording.
    """
    START = 1
    STOP = 2


def on_beat(current_frame: int, len_beat: int, frames: int) -> bool:
    """Determine whether beat will happen in next frames.
    Returns:
        bool: True if beat happens, False if it does not.
    """
    position_in_beat = current_frame % len_beat
    if position_in_beat+frames >= len_beat:
        return True
    return False


def is_in_first_half_of_beat(current_frame: int, len_beat: int) -> bool:
    """Determine whether current_frame is in the first half of beat. 
    In case it is exactly in half, return True.

    Args:
        current_frame (int): Current frame (effectively moment in time).
        len_beat (int): The length of one beat in frames.

    Returns:
        bool: True if current_frame is in the first half of beat.
    """
    half_beat = int(len_beat/2)
    position_in_beat = current_frame % len_beat
    if position_in_beat <= half_beat:
        return True
    return False

