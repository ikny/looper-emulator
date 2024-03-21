from abc import ABC, abstractmethod


class AbstractLemApp(ABC):
    """AbstractLemApp is a contract for LemApp class, written so that various GUI widgets 
    can be given a guarantee (through typing) that their parent has a specific method. """
    @abstractmethod
    def set_bpm(self, bpm: int) -> None:
        ...

    @abstractmethod
    def show_err(self, message: str) -> None:
        ...

    @abstractmethod
    def destroy(self) -> None:
        ...

    @abstractmethod
    def on_start_recording(self) -> None:
        ...

    @abstractmethod
    def on_stop_recording(self) -> None:
        ...

    @abstractmethod
    def delete_track(self, idx: int) -> None:
        ...
