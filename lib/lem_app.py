# types/dev
from typing import Optional
import logging
# libs
import tkinter as tk
from soundfile import LibsndfileError
import sys
# parts of project
from .abstract_lem_app import AbstractLemApp
from .gui_classes import AppBar, ErrorPopup, RecordButton, TrackList
from .lem import Lem
from .constants import METRONOME_SAMPLE_PATH, SAMPLERATE
from .custom_exceptions import InvalidSamplerateError


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(levelname)s: %(asctime)s %(name)s: %(message)s", level=logging.DEBUG)


class LemApp(tk.Tk, AbstractLemApp):
    """The main class of this app. Provides a GUI and creates its own state.
    """

    def __init__(self, screenName: str | None = None, baseName: str | None = None, className: str = "Tk", useTk: bool = True, sync: bool = False, use: str | None = None) -> None:
        """Initializes a new instance of LemApp and start running.

        Args (from tkinter documentation):
            Return a new Toplevel widget on screen SCREENNAME. A new Tcl interpreter will
            be created. BASENAME will be used for the identification of the profile file (see
            readprofile).
            It is constructed from sys.argv[0] without extensions if None is given. CLASSNAME
            is the name of the widget class.
        """
        super().__init__(screenName, baseName, className, useTk, sync, use)

        self.lem_state: Optional[Lem] = None

        # set GUI to darkmode
        self.tk_setPalette(background='#181818', foreground='#DDD78D')

        self.title("lem Looper Emulator")

        # create GUI elements
        self.app_bar = AppBar(master=self)
        self.app_bar.pack()
        self.record_button = RecordButton(
            master=self, state="disabled", on_start_recording=self.on_start_recording, on_stop_recording=self.on_stop_recording)
        self.record_button.pack(fill="x", expand=0)
        self.tracklist = TrackList(master=self)
        self.tracklist.pack(fill="both", expand=1)

        # start running
        self.mainloop()

    def set_bpm(self, bpm: int) -> None:
        """Tries to initialize the logic (Lem). If everything succeeds, prepares the GUI for the actual recording flow. 

        Args:
            bpm (int): The value the user has entered into BpmPopup.
        """
        try:
            self.lem_state = Lem(bpm=bpm)
        except LibsndfileError:
            self.show_err(
                message=f"""
                The file on specified path could not be opened. 
                Please check that path "{METRONOME_SAMPLE_PATH}" contains valid audio file.""")
            return
        except InvalidSamplerateError:
            self.show_err(
                message=f"""
                The samplerate of the provided metronome sample does not match the required samplerate ({SAMPLERATE}).
                Please provide a valid audio file.""")
        except Exception as e:
            self.show_err(f"An unexpected error occured: \n{e}")
            return

        self.app_bar.update_bpm(bpm=bpm)
        self.record_button["state"] = "normal"

    def show_err(self, message: str) -> None:
        """Build an error popup with a message.

        Args:
            message (str): The message to be shown to the user.
        """
        ErrorPopup(master=self, message=message)

    def destroy(self) -> None:
        """A method to be called when the user closes the app's main window. Ensures the state has terminated.
        """
        if self.lem_state:
            self.lem_state.terminate()
        return super().destroy()

    """ The following methods are here because various GUI elements use them. """

    def on_start_recording(self) -> None:
        """A method to be passed as a callback to the RecordButton. Delegates the action to its state object.
        """
        logger.debug("Recording button pushed, starting recording!")
        self.lem_state.start_recording()  # type: ignore

    def on_stop_recording(self) -> None:
        """A method to be passed as a callback to the RecordButton.
        Adds track in the GUI only if the track was added in the logic (see LoopStreamManager.post_production for more details).
        """
        logger.debug("Recording button pushed, stopping recording!")
        if self.lem_state.stop_recording():  # type: ignore
            self.tracklist.add_track()

    def delete_track(self, idx: int) -> None:
        """Delegates the track deletion to the state.

        Args:
            idx (int): The index of the track which is being deleted.
        """
        self.lem_state.delete_track(idx=idx)  # type: ignore


if __name__ == "__main__":
    app = LemApp()
