# YOLO training related helpers
from pathlib import Path
from tqdm import tqdm  
import json
from datetime import datetime
import numpy as np  
from zarr.core import array # For type checking 
from octron.sam2_octron.helpers.sam2_zarr import load_image_zarr   
from octron.yolo_octron.helpers.polygons import get_polygons


def load_object_organizer(file_path):
    """
    Load object organizer .json from disk and return
    its content as dictionary.
    The .json file itself has been created via the save_to_disk method in 
    the object_organizer class (octron.sam2_octron.object_organizer.py).
    
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
                         polygons (dict) # Polygons for each frame index
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

        polys = {} # Collected polygons over frames
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
            polys[frame] = mask_polys_
        labels[label_id]['polygons'] = polys  

    return labels


def draw_polygons(labels, 
                  video_data,
                  max_to_plot=2
                  ):   
    """
    Helper.
    Draw the polygons on the video frames, after having created the labels
    dictionary via the collect_labels() function.

    Parameters
    ----------
    labels : dict : Dictionary containing labels and their corresponding frames
            keys: label_id
            values: dict
                keys: label, frames, masks, polygons, color
                values: label (str), # Name of the label
                         frames (np.array), # Annotated frame indices for the label
                         masks (list of zarr arrays), # Masks for each frame
                         polygons (dict) # Polygons for each frame index
                         color (list) # Color of the label (RGBA, [0,1])   
    video_data : np.array : Video data array
    max_to_plot : int : Maximum number of frames to plot per label
    
    
    """
    # Check if cv2 is installed correctly
    try:
        import cv2 
    except ModuleNotFoundError:
        print('Please install cv2 first, via pip install opencv-python')
        return
    # ... and matplotlib
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print('Please install matplotlib first, via pip install matplotlib')
        return

    if max_to_plot < 1:
        max_to_plot = 1
    print(f'Drawing polygons for {len(labels)} labels.')
    print(f'{max_to_plot} frame(s) per label max. will be plotted.')
    # Draw the polygons on the video frames
    for label_id in labels:
        label = labels[label_id]['label']
        frames = labels[label_id]['frames']
        color = np.array(labels[label_id]['color'])[:-1]*255
        counter = 1
        for frame in frames:
            
            current_frame = video_data[frame].copy()
            polys = labels[label_id]['polygons'][frame]  
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
            if counter >= max_to_plot:
                break   
            counter += 1
            
            
def train_test_val(frame_indices,
                   training_fraction=0.8,
                   validation_fraction=0.1,
                   random_seed=88,
                   verbose=False,
                   ):
    """
    Perform a train-test-validation split on the frame indices.
    
    Parameters
    ----------
    frame_indices : np.array : Array of frame indices
    training_fraction : float : Fraction of training data
    validation_fraction : float : Fraction of validation data
    random_seed : int : Random seed for reproducibility
    verbose : bool : Whether to print sizes of the splits
    
    Returns
    -------
    split_dict : dict : Dictionary containing the splits
        keys: 'train', 'val', 'test'
        values: np.array : Frame indices for each split
    
    """
    assert training_fraction + validation_fraction < 1, 'Fractions should sum to less than 1'
    assert training_fraction > validation_fraction, 'Training fraction should be greater than validation fraction'
    assert len(frame_indices) > 0, 'No frame indices provided'
    
    # Shuffle the indices
    np.random.seed(random_seed)
    shuffled_indices = np.random.permutation(len(frame_indices))

    # Calculate split points
    train_size = int(training_fraction * len(frame_indices))
    val_size = int(validation_fraction * len(frame_indices))
    # test_size = len(frames) - train_size - val_size (remaining frames)

    # Split the shuffled indices
    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:train_size+val_size]
    test_indices = shuffled_indices[train_size+val_size:]

    # Get the actual frame numbers for each split
    train_frames = frame_indices[train_indices]
    val_frames = frame_indices[val_indices]
    test_frames = frame_indices[test_indices]

    if verbose:
        # Print sizes
        print(f"Total frames: {len(frame_indices)}")
        print(f"Training set: {len(train_frames)} frames")
        print(f"Validation set: {len(val_frames)} frames")
        print(f"Test set: {len(test_frames)} frames")
        
    split_dict = {
        'train': train_frames,
        'val': val_frames,
        'test': test_frames
    }

    # Sanity check 
    collected_frames = []
    for frames in split_dict.values():
        assert len(frames) > 0, 'Empty split found'
        collected_frames.extend(frames)
    assert len(collected_frames) == len(frame_indices), 'Not all frames were collected in the splits'
    assert set(collected_frames) == set(frame_indices), 'Some frames are missing in the splits'

    return split_dict


def write_training_data(labels,
                        path_to_training_root,
                        video_data,
                        ):
    
    try:
        from PIL import Image
    except ModuleNotFoundError:
        print('Please install PIL first, via pip install pillow')
        return
    
    for label_id, label_dict in tqdm(labels.items(), desc=f'Exporting {len(labels)} labels'):
        # Extract the size of the masks for normalization later on 
        for m in label_dict['masks']:
            assert m.shape == label_dict['masks'][0].shape, f'All masks should have the same shape'
        _, mask_height, mask_width = label_dict['masks'][0].shape
        
        for split in ['train', 'val', 'test']:
            for frame_id in tqdm(label_dict['frames_split'][split], 
                                 desc=f'Exporting {split} frames', 
                                 leave=False
                                 ):
                frame = video_data[frame_id]
                image_output_path = path_to_training_root / split / f'frame_{frame_id}.png'
                if not image_output_path.exists():
                    # Convert to uint8 if needed
                    if frame.dtype != np.uint8:
                        if frame.max() <= 1.0:
                            frame_uint8 = (frame * 255).astype(np.uint8)
                        else:
                            frame_uint8 = frame.astype(np.uint8)
                    else:
                        frame_uint8 = frame
                    # Convert to PIL Image
                    img = Image.fromarray(frame_uint8)
                    # Save with specific options for higher quality
                    img.save(
                        image_output_path,
                        format="PNG",
                        compress_level=0,  # 0-9, lower means higher quality
                        optimize=True,
                    )
                
                # Create the label text file with the correct format
                with open(path_to_training_root / split / f'frame_{frame_id}.txt', 'a') as f:
                    for polygon in label_dict['polygons'][frame_id]:
                        f.write(f'{label_id}')
                        # Write each coordinate pair as normalized coordinate to txt
                        for point in polygon:
                            f.write(f' {point[0]/mask_height} {point[1]/mask_width}')
                        f.write('\n')
    print(f"Training data exported to {path_to_training_root}")

def write_yolo_config_yaml(
    output_path: Path,
    dataset_path: Path,
    train_path: str,
    val_path: str,
    test_path: str,
    label_dict: dict
    ):
    """
    Write a YOLO configuration YAML file to disk.
    
    Parameters
    ----------
    output_path : Path, str
        Path where to save the YAML file
    dataset_path : Path, str
        Path to the dataset root directory
    train_path : str
        Folder name of the training images (relative to dataset_path)
    val_path : str
        Folder name of the validation images (relative to dataset_path)
    test_path : str, optional
        Folder name of the images (relative to dataset_path)
    label_dict : dict, optional
        Dictionary mapping label IDs to label names
    """
    import yaml
    assert output_path.suffix == '.yaml', 'Output file should have a .yaml extension'
    assert dataset_path.exists(), f'Dataset path not found at {dataset_path}'
    assert dataset_path.is_dir(), f'Dataset path should be a directory, but found a file at {dataset_path}' 
    
    # Create the config dictionary
    config = {
        "path": str(dataset_path),
        "train": train_path,
        "val": val_path,
    }
    
    if test_path:
        config["test"] = test_path
    
    # Add class names if provided
    if label_dict:
        config["names"] = label_dict
    
    # Write header comments
    header = "# OCTRON training dataset\n# Exported on {}\n\n".format(datetime.now())
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(header)
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"YOLO config saved to {output_path}")
    return output_path


