import importlib.metadata
from importlib.metadata import version

try:
    __version__ = version("octron")
except importlib.metadata.PackageNotFoundError:
    __version__ = "no version"  

from .main import octron_widget
from .reader import octron_reader
from .yolo_octron.helpers.yolo_results import YOLO_results

__all__ = (
    "octron_widget",
    "octron_reader",
    "YOLO_results",
)

