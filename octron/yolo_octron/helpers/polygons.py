# Polygon helpers for training data extraction

import numpy as np  

def find_objects_in_mask(mask, min_area=10):
    """
    Find all objects in a binary mask using connected component labeling.
    This is run initially to gain an understanding of the objects in the masks,
    i.e. know their median area, etc.
    See also:
    https://scikit-image.org/docs/0.23.x/api/skimage.feature.html#
    
    
    Parameters:
    -----------
    mask : numpy.ndarray
        Binary mask where objects have value 1 
        and background has value 0
    min_area : int
        Minimum area (in pixels) for an object to be considered
        
    Returns:
    --------
    labels : numpy.ndarray
        Labeled image where each object has a unique integer value
    regions : list
        List of region properties for each object
    """
    try:
        from skimage import measure
    except ImportError:
        raise ImportError('find_objects_in_mask() requires scikit-image')
    
    # Ensure the mask is binary
    binary_mask = mask > 0
    
    # Label connected components
    labels = measure.label(binary_mask, background=0, connectivity=2)
    
    # Get region properties
    regions = measure.regionprops(labels)
    
    # Filter regions by size if needed
    if min_area > 0:
        for region in regions[:]:
            if region.area < min_area:
                # Remove small objects
                labels[labels == region.label] = 0
                regions.remove(region)
    
    # Relabel the image to ensure consecutive labels
    if any(region.label != i+1 for i, region in enumerate(regions)):
        new_labels = np.zeros_like(labels)
        for i, region in enumerate(regions):
            new_labels[labels == region.label] = i + 1
        labels = new_labels
        regions = measure.regionprops(labels)
    
    return labels, regions



def watershed_mask(mask,
                   footprint_diameter,
                   min_size_ratio=0.1,
                   plot=False,
                   ):
    """
    Watershed segmentation of a mask image
    
    Parameters
    ----------
    mask : np.array : Binary mask where objects have value 1 
                      and background has value 0
    footprint_diameter : float : Diameter of the footprint for peak_local_max()
    min_size_ratio : float : Minimum size ratio of the largest mask to keep a mask
    plot : bool : Whether to plot the results
    
    
    Returns
    -------
    labels : np.array : Segmented mask, where 0 is background 
                        and each object has a unique integer value
    masks : list : List of binary masks for each object
    
    
    """
    try:
        from scipy import ndimage as ndi
    except ImportError:
        raise ImportError('watershed_mask() requires scipy')
    try:
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_max
    except ImportError:
        raise ImportError('watershed_mask() requires scikit-image')
    
    assert mask.ndim == 2, f'Mask should be 2D, but got ndim={mask.ndim}'
    assert not np.isnan(mask).any(), 'There are NaNs in input mask' # If this happens, it can be solved!
    assert set(np.unique(mask)) == set([1,0]), 'Mask should be composed of 0s and 1s'
    
    diam = int(np.round(footprint_diameter))
    assert diam > 0, 'Footprint diameter should be a positive integer'
    
    # Watershed segmentation
    # See https://scikit-image.org/docs/0.24.x/auto_examples/segmentation/plot_watershed.html
    distance = ndi.distance_transform_edt(mask)
    diam = int(np.round(diam))
    peak_dist = int(np.round(diam/2))
    coords = peak_local_max(distance, 
                            footprint=np.ones((diam,diam)), 
                            labels=mask,
                            min_distance=peak_dist,
                            p_norm=2, # Euclidean distance
                            )
    mask_ = np.zeros(distance.shape, dtype=bool)
    mask_[tuple(coords.T)] = True
    markers, _ = ndi.label(mask_)
    labels = watershed(-distance, markers, mask=mask) # this is the segmentation
            
    masks = []
    areas = []
    for l in np.unique(labels):
        if l == 0:  
            # That's background
            continue
        labelmask = np.zeros_like(labels)
        labelmask[labels == l] = 1
        masks.append(labelmask)    
        areas.append(np.sum(labelmask))
        
    # If we have more than one mask, check for size disparities
    # Filter out masks that are too small
    if len(masks) > 1:
        max_area = max(areas)
        filtered_masks = []
        for mask_idx, area in enumerate(areas):
            # Keep the mask if it's at least min_size_ratio of the largest mask
            if area >= min_size_ratio * max_area:
                filtered_masks.append(masks[mask_idx])
        masks = filtered_masks
        
    # Create a new labels image
    labels = np.zeros_like(mask)
    for i, m in enumerate(masks):
        labels[m] = i + 1

    if plot:
        import matplotlib.pyplot as plt
        plt.rcParams['xtick.major.size'] = 10
        plt.rcParams['xtick.major.width'] = 1
        plt.rcParams['ytick.major.size'] = 10
        plt.rcParams['ytick.major.width'] = 1
        plt.rcParams['xtick.bottom'] = True
        plt.rcParams['ytick.left'] = True

        _, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(mask, cmap='gray')
        ax[0].set_title('Original mask')
        ax[1].imshow(labels, cmap='nipy_spectral')
        ax[1].set_title(f'Watershed, remaining masks: {len(masks)}')
        plt.show()
        
    return labels, masks


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
    try:
        from imantics import Mask
    except ImportError:
        raise ImportError('get_polygons() requires imantics')
    
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
