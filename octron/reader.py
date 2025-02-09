from pathlib import Path
from typing import Union, Sequence, Callable, List, Optional
from napari.types import LayerData
from napari.utils.notifications import (
    show_error,
)
# Define some types
PathLike = str
PathOrPaths = Union[PathLike, Sequence[PathLike]]
ReaderFunction = Callable[[PathOrPaths], List[LayerData]]

import warnings
warnings.simplefilter("once")


def octron_reader(path: "PathOrPaths") -> Optional["ReaderFunction"]:
    """
    OCTRON napari reader.
    Accepts OCTRON project folders.
    
    Parameters
    ----------
    path : str or list of str
        Path to a file or folder.
    
    Returns
    -------
    function : Callable
        Function to read the file or folder.
        
    """
    
    path = Path(path)
    if path.is_dir() and path.exists():
        return read_octron_folder
        
    if path.is_file() and path.exists():
        return read_octron_file

def read_octron_file(path: "PathOrPaths") -> List["LayerData"]:
    """
    Single file reads that are dropped in the main window are not supported.
    """
    show_error(
        f"Single file drops to main window are not supported"
    )
    return [(None,)]

def read_octron_folder(path: "PathOrPaths") -> List["LayerData"]:
    print(
        f"🐙 Checking putative OCTRON project folder: {path}"
    )
    # Check if the folder contains the necessary files
    
    
    
    
    
    return [(None,)]


