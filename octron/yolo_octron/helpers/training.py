# YOLO training related helpers
from pathlib import Path
from tqdm import tqdm  
import json
import numpy as np  
from zarr.core import array # For type checking 
from octron.sam2_octron.helpers.sam2_zarr import load_image_zarr   
from octron.yolo_octron.helpers.polygons import get_polygons


def load_object_organizer(file_path):
    """
    Load object organizer .json from disk and return
    its content as dictionary.
    The .json file has been created via the save_to_disk method.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the .json file.
    
    Returns
    -------
    dict
        Dictionary containing all object organizer data.
    
    """
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"No organizer file found at {file_path}")
        return
    if not file_path.suffix == '.json': 
        print(f"❌ File is not a json file: {file_path}")
        return
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"📖 Octron object organizer loaded from {file_path.as_posix()}")
        return data
    except Exception as e:
        print(f"❌ Error loading json: {e}")
        return 


def collect_labels(organizer_dict,
                   expected_num_frames,
                   expected_image_height,
                   expected_image_width,
                  ):
    """
    Create a dictionary of labels and their corresponding frames.   
    Extract polygons for all labels along the way.
    
    Performs sanity checks on the data and ensures that the data is consistent
    across the object organizer, zarr data and video data in terms of 
    number of frames, image height and image width.
    
    Parameters
    ----------
    organizer_dict : dict : Dictionary containing OCTRON organizer data.
        This is the json data loaded via load_object_organizer() function.
        It contains label ID and names as well as paths to the zarr files that 
        contain the mask data.
    expected_num_frames : int : Expected number of frames in the video
    expected_image_height : int : Expected height of the image (video data)
    expected_image_width : int : Expected width of the image (video data)

    Returns
    -------
    labels : dict : Dictionary containing labels and their corresponding frames
            keys: label_id
            values: dict
                keys: label, frames, masks, polygons, color
                values: label (str), # Name of the label
                         frames (np.array), # Annotated frame indices for the label
                         masks (list of zarr arrays), # Mask zarr arrays
                         polygons (list of np.arrays) # Polygons for each frame
                         color (list) # Color of the label (RGBA, [0,1])   
                                
    """
    labels = {}
    for entry in organizer_dict['entries'].values():
        label_id = int(entry['label_id'] )
        label    = entry['label'] 
        color    = entry['color']
        if label_id in labels:
            assert labels[label_id]['label'] == label, 'Label name vs. id do not match'
        else:
            # First time we see this label
            labels[label_id] = {'label':label, 
                                'frames': [],
                                'masks': [], 
                                'color': color, # Only save the first color
                                }   

        # Find out which frames were annotated
        zarr_path = Path(entry['prediction_layer_metadata']['zarr_path'])
        assert zarr_path.exists(), f'Zarr file not found at {zarr_path}'
        num_frames, image_height, image_width = entry['prediction_layer_metadata']['data_shape']
        loaded_masks, status = load_image_zarr(zarr_path, 
                                    num_frames,
                                    image_height, 
                                    image_width, 
                                    num_ch=None,
                                    )
        assert status == True
        assert loaded_masks is not None
        assert isinstance(loaded_masks, array.Array), f'Expected zarr array for masks, got {type(loaded_masks)}'
        
        # Comparisons
        # Make sure information is consistent across object organizer (json),
        # zarr data and video data
        # assert data organizer == data video data == data zarr array
        assert num_frames == expected_num_frames == loaded_masks.shape[0]
        assert image_height == expected_image_height == loaded_masks.shape[1]
        assert image_width == expected_image_width == loaded_masks.shape[2]
        
        labels[label_id]['masks'].append(loaded_masks) # This is the zarr array
        # Extract annotated frame indices
        # The fill value of the zarr array is -1, so we can use this to find annotated frames
        annotated_indices = np.where(loaded_masks[:,0,0] >= 0)[0]
        labels[label_id]['frames'].extend(annotated_indices) 
        
    # Maintain only unique entries in 'frames' lists
    for label_id in labels:
        _, i = np.unique(labels[label_id]['frames'], return_index=True)
        labels[label_id]['frames'] = np.array(labels[label_id]['frames'])[np.sort(i)]
        print(f'Label {labels[label_id]["label"]} has {len(labels[label_id]["frames"])} annotated frames')
    
    # Extract polygon points for each label
    for label_id in labels:
        label = labels[label_id]['label']
        frames = labels[label_id]['frames']

        polys = [] # Collected polygons over frames
        for frame in tqdm(frames, desc=f'Polygons for label {label}'):    
            mask_polys_ = [] # List of polygons for the current frame
            for mask_zarr in labels[label_id]['masks']:
                mask_ = mask_zarr[frame]
                try:
                    mask_polys_.append(get_polygons(mask_)) 
                except AssertionError:
                    # The mask is empty at this frame.
                    # This happens if there is more than one mask 
                    # zarr array (because there are multiple instances of a label), 
                    # and the current label is not present in the current mask array.
                    pass    
            polys.append(mask_polys_) 
        labels[label_id]['polygons'] = polys  

    return labels


def draw_polygons(labels, video_data, max_to_plot=2):   
    # Check if cv2 is installed correctly
    try:
        import cv2 
    except ModuleNotFoundError:
        print('Please install cv2 first, via pip install opencv-python')
        return

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print('Please install matplotlib first, via pip install matplotlib')
        return

    """
    Draw the polygons on the video frames.

    Parameters
    ----------
    labels : dict : Dictionary containing labels and their corresponding frames
            keys: label_id
            values: dict
                keys: label, frames, masks, polygons, color
                values: label (str), # Name of the label
                         frames (np.array), # Annotated frame indices for the label
                         masks (list of zarr arrays), # Masks for each frame
                         polygons (list of np.arrays) # Polygons for each frame
                         color (list) # Color of the label (RGBA, [0,1])   
    video_data : np.array : Video data array
    max_to_plot : int : Maximum number of frames to plot per label
    
    
    """
    if max_to_plot < 1:
        max_to_plot = 1
    print(f'Drawing polygons for {len(labels)} labels.')
    print(f'{max_to_plot} frame(s) per label max. will be plotted.')
    # Draw the polygons on the video frames
    for label_id in labels:
        label = labels[label_id]['label']
        frames = labels[label_id]['frames']
        color = np.array(labels[label_id]['color'])[:-1]*255
        for no, frame in enumerate(frames):
            
            current_frame = video_data[frame].copy()
            polys = labels[label_id]['polygons'][no]  
            frame_and_polys = cv2.polylines(img=current_frame, 
                                            pts=polys, 
                                            isClosed=True, 
                                            color=color.tolist(), 
                                            thickness=5,
                                            )
            
            # Draw
            _, ax = plt.subplots(1,1)    
            ax.imshow(frame_and_polys)
            ax.set_xticks([])   
            ax.set_yticks([])
            ax.set_title(f'Label: "{label}" - frame {frame}')
            plt.show()   
            if no >= max_to_plot-1:
                break   