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
    path = Path(path)
    # Check what kind of folder you are dealing with.
    # There are three options:
    # A. Octron project folder
    # B. Octron video (annotation) folder
    # C. Octron prediction (results) folder
    
    # Case A 
    
    
    
    
    
    
    # Case C 
    # Check if the folder has .csv files AND a predictions.zarr 
    csvs = list(path.glob("*.csv"))
    prediction_zarr = list(path.glob("predictions.zarr"))
    if csvs and prediction_zarr:
        print(
            f"🐙 Detected OCTRON prediction folder: {path}"
        )
        # Load predictions
        from octron.yolo_octron.yolo_octron import YOLO_octron
        yolo_octron = YOLO_octron()
        yolo_octron.show_predictions(
            save_dir = path,
            sigma_tracking_pos = 2, # Fixed for now 
        )
        return [(None,)]
    
    
    return [(None,)]


