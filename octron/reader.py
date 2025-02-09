from pathlib import Path
import napari
import warnings
from octron.sam2_octron.helpers.video_loader import probe_video, get_vfile_hash
from napari_pyav._reader import FastVideoReader
warnings.simplefilter("once")


def octron_reader(path):
    """
    OCTRON napari reader.
    Accepts OCTRON project folders.
    

    
    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    out : list of tuple
        List of tuples containing the video reader object, 
        metadata dictionary and the layer type.
    """
    paths = [Path(path)] if isinstance(path, str) else path
    if len(paths) > 1:
        return None
    
    # Folder or video file ? 
    path = paths[0]
    if path.is_dir() and path.exists():
        return read_octron_folder
        
    if path.is_file():
        return None


def read_octron_folder(path):
    path = Path(path)
    print(
        f"🐙 Checking putative OCTRON project folder: {path}"
    )
    return None


