from pathlib import Path
from napari.utils.notifications import show_info
import warnings
from octron.sam2_octron.helpers.video_loader import probe_video, get_vfile_hash
from napari_pyav._reader import FastVideoReader
warnings.simplefilter("once")

ACCEPTED_CONTAINERS = ['.mp4'] # Strictly speaking, only mp4 should be allowed 






def octron_reader(path):
    """
    OCTRON napari reader.
    Accepts both OCTRON project folders, 
    as well as video files in the accepted formats.
    (see ACCEPTED_CONTAINERS)
    

    
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
        show_info("Only one file or folder at a time is supported.")
        return None
    
    # Folder or video file ? 
    
    path = paths[0]
    if path.is_dir() and path.exists():
        show_info(f"Folder {path.as_posix()}")
        return read_octron_folder
        
    if path.is_file() and path.exists():
        if not path.suffix in ACCEPTED_CONTAINERS:
            print(f"File {path} is not a video file. Accepted formats are {ACCEPTED_CONTAINERS}")
            return None
        else:
            return read_octron_video


def read_octron_folder(path):
    path = Path(path)
    return

def read_octron_video(path):
    path = Path(path)
    video_dict = probe_video(path)
    video_file_hash = get_vfile_hash(path)
    video_dict['hash'] = video_file_hash
    layer_dict = {'name'    : f'VIDEO [name: {path.stem}]',
                  'metadata': video_dict,
                  }
    
    out = [(FastVideoReader(path, read_format='rgb24'), layer_dict , 'image')]
    return out