import hashlib
from pathlib import Path
from . import sam2_checks
from . import build_sam2_octron
from . import sam2_octron
from . import sam2_colors
from . import sam2_zarr
from . import sam2_mask_layer
from . import sam2_points_layer


def get_file_hash(filepath, block_size=65536):
    '''
    Get fast hash digest of file using blake2b. 
    
    Parameters
    ----------
    filepath : str or Path
        Path to file.
    block_size : int
        Number of bytes to read at a time.
        
    Returns
    -------
    str : Hash digest.
    
    '''
    filepath = Path(filepath)
    assert filepath.exists(), f'File does not exist: {filepath}'
    
    hasher = hashlib.blake2b(digest_size=32)
    with open(filepath, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            hasher.update(block)
    return hasher.hexdigest()