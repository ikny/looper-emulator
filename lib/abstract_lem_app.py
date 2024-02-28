from abc import ABC

"""TODO: This is a dirty workaround againts circular dependency LemApp <-> gui_classes.
However, if this class defined the contract properly, I guess it would be clean.
"""


class AbstractLemApp(ABC):
    ...
