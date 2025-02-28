# Polygon helpers for training data extraction

import numpy as np  
from imantics import Mask

def get_polygons(mask):
    """
    Given a mask image, extract outlines as polygon 
    
    Parameters
    ----------
    mask : np.array : Mask image, composed of 0s and 1s, where 1 

    Returns
    -------
    polygon_points : np.array : Polygon points for the extracted binary mask(s)
    
    """
    if mask is None:
        return None 
    
    assert mask.ndim == 2, f'Image should be 2D, but got ndim={mask.ndim}'
    assert not np.isnan(mask).any(), 'There are NaNs in input image' # If this happens, it can be solved!
    assert set(np.unique(mask)) == set([1,0]), 'Image should be composed of 0s and 1s'   
    
    polygons = Mask(mask).polygons()

    if len(polygons.points) > 1:
        # Just fuse the lists
        polygon_points = np.concatenate(polygons.points,axis=0)
    else:
        polygon_points = polygons.points[0]
        
    return polygon_points
