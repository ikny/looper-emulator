import tkinter as tk
from typing import Any, Callable

from lem import Lem
from main import LemApp


class BpmPopup(tk.Toplevel):
    """A GUI class representing a dialog window. The dialog asks for BPM value and checks its validity. If valid, sets BPM.
    """

    def __init__(self, master: LemApp, **kwargs: Any) -> None:
        """Build a new BpmPopup.

        Args:
            master (LemApp): The parent widget. Must be an instance of Lem class (the top level gui class of this project).
        """
        super().__init__(master, **kwargs)
        self.master: LemApp = master

        self.title("enter BPM")

        self._entry_line = tk.Frame(master=self)
        self._message = tk.Label(
            master=self._entry_line, text="Insert the desired BPM value: ")
        self._bpm_entry = tk.Entry(master=self._entry_line, width=3)

        self._message.pack(side="left")
        self._bpm_entry.pack(side="right")
        self._entry_line.pack(side="top", padx=30, pady=5)

        self._instructions = tk.Label(
            master=self, text="BPM value must be a whole number between 1 and 400")
        self._instructions.pack()

        self._confirm = tk.Button(
            master=self, text="Confirm!", command=self.set_bpm)
        self._confirm.pack(side="bottom", padx=5, pady=5)

    def validate(self, value: str) -> bool:
        """A method evaluating whether the value meets the conditions for suitable BPM (an integer between 1 and 400)

        Args:
            value (str): value taken from an entry

        Returns:
            bool: True if value meets the conditions, else False.
        """
        try:
            val = int(value)
        except ValueError:
            return False

        return val > 0 and val <= 400

    def set_bpm(self) -> None:
        """Try whether the entry value is a suitable BPM value. If yes, set it as a BPM.
        """
        entry_value = self._bpm_entry.get()
        if not self.validate(value=entry_value):
            return
        bpm = int(entry_value)
        self.master.set_bpm(bpm=bpm)
        self.destroy()


class AppBar(tk.Frame):
    """A GUI class representing an app bar (the topmost widget on the page). 
    Contains label displaying the set BPM, and initially also a button enabling to set it. 
    This button is destroyed afterwards.
    """

    def __init__(self, state: Lem, master: LemApp, **kwargs: Any) -> None:
        """Build a new AppBar.

        Args:
            state (Lem): The main state class, managing all the logic.
            master (LemApp): The parent widget. Must be an instance of Lem class (the top level gui class of this project).
        """
        super().__init__(master, **kwargs)
        self.master: LemApp = master

        self._bpm_lbl = tk.Label(master=self, text="BPM: ")
        self._bpm_lbl.pack(side="left", padx=5, pady=5)

        self._dialog_button = tk.Button(
            master=self, text="set BPM", command=self.invoke_dialog)
        self._dialog_button.pack(side="right", padx=5, pady=5)

    def update_bpm(self, bpm: int) -> None:
        """Update the BPM label and destroy the dialog button.

        Args:
            bpm (int): The BPM value.
        """
        self._bpm_lbl.config(text=f"BPM: {bpm}")
        self._dialog_button.destroy()

    def invoke_dialog(self) -> None:
        """Open a popup dialog.
        """
        BpmPopup(master=self.master)


class RecordButton(tk.Button):
    """A GUI class representing a two-state button through which users can start or stop recording of tracks.
    """

    def __init__(self, master: tk.Misc, start_recording: Callable[[], None], stop_recording: Callable[[], None], **kwargs: Any) -> None:
        """Build a new RecordButton.

        Args:
            master (tk.Misc): The parent widget.
            start_recording (function): The callback to be called when the state changes to "recording" (on odd pushes).
            stop_recording (function): The callback to be called when the state changes to "waiting" (on even pushes).
        """
        super().__init__(master, text="Press to start recording (or press SPACE)", height=2,
                         command=self._clicked, **kwargs)
        self.start_recording = start_recording
        self.stop_recording = stop_recording
        self._state = "waiting"

    def _clicked(self) -> None:
        """A method to be called when the button is pushed. 
        Based on the current state decides what callback should be called and changes its state.
        """
        if self._state == "waiting":
            self.start_recording()
            self._state = "recording"
            self.config(text="Press to stop recording (or press SPACE)")
        else:
            self.stop_recording()
            self._state = "waiting"
            self.config(text="Press to start recording (or press SPACE)")


class TrackList(tk.Frame):
    """A GUI class representing a scrollable list of recorded tracks.
    """

    def __init__(self, master: LemApp, **kwargs: Any) -> None:
        """Build a new Tracklist.

        Args:
            master (LemApp): The parent widget. Must be an instance of Lem class (the top level gui class of this project).
        """
        super().__init__(master, **kwargs)
        self.master: LemApp = master

        # {track_id: Track}
        self._tracks: dict[int, Track] = {}
        self._free_id = 0

        # using tk.Canvas widget, because tk.Listbox only works for text
        self._scrollable = tk.Canvas(master=self, height=200, width=280)
        self._scrollable.pack(side="left", fill="both", expand=1)

        self._track_frame = tk.Frame(master=self._scrollable)
        self._track_frame.pack(side="left", fill="both", expand=1)

        self._scroller = tk.Scrollbar(
            master=self, command=self._scrollable.yview)
        self._scroller.pack(side="right", fill="y", expand=0)

        # setup the widgets so the scrolling works
        self._scrollable.create_window(
            (0, 0), window=self._track_frame, anchor="nw")
        self._scrollable.config(yscrollcommand=self._scroller.set)

    def add_track(self) -> None:
        """Create a new Track instance and call the _update_sizes function, which updates the scrollable region.
        """
        track = Track(id=self._free_id,
                      master=self._track_frame, tracklist=self)
        self._tracks[self._free_id] = track
        self._free_id += 1
        track.pack(fill="both", expand=1, pady=1)
        self._update_sizes()

    def delete_track(self, track_id: int) -> None:
        """Deletes a reference to the track with the provided ID. 
        Finds out the index of the Track, because in Lem (the state) they are stored in a list.

        Args:
            track_id (int): The unique ID of the Track. 
        """
        keys = self._tracks.keys()
        track_indexes = list(keys)
        track_index = track_indexes.index(track_id)
        # TODO: this is pretty tightly coupled, idk what solution would be cleaner.
        self.master.lem_state.delete_track(idx=track_index)

        self._tracks.pop(track_id)
        self._update_sizes()

    def _update_sizes(self) -> None:
        """Updates the size of the scrollable region and of the frame because a Track was added/deleted.
        """
        self._track_frame.update()
        self._scrollable.config(scrollregion=(
            0, 0, 0, self._track_frame.winfo_height()))


class Track(tk.Frame):
    """A GUI class to represent a recorded track.
    Includes a label with the track ID, so the user can identify the track, and a delete button.
    """

    def __init__(self, id: int, master: tk.Frame, tracklist: TrackList, **kwargs: Any) -> None:
        """Build a new Track.

        Args:
            id (int): The unique ID of the track.
            master (tk.Frame): The parent widget. 
            tracklist (TrackList): The Tracklist widget, which acts as a manager for the Tracks.
        """
        # add border to the track frame
        super().__init__(master, highlightbackground="#DDD78D", highlightthickness=1, **kwargs)

        self._tracklist = tracklist

        self._track_id = id
        self.name = tk.Label(master=self, text=f"track {id}")
        self.name.pack(side="left", padx=10, pady=10)

        image = tk.PhotoImage(file="lib/images/trash-bin.png")
        # Make the image smaller. In case of using an image of a different size,
        # the argument to the subsample function should be changed accordingly.
        self._image = image.subsample(10)

        self._delete_button = tk.Button(
            master=self, image=self._image, command=self.destroy)
        self._delete_button.pack(side="right")

    def destroy(self) -> None:
        """Calls the delete_track method of self._tracklist, and then destroys itself.
        """
        self._tracklist.delete_track(track_id=self._track_id)
        return super().destroy()
