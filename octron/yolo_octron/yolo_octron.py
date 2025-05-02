# Main YOLO Octron class
# We are using YOLO11 as the base class for YOLO Octron.
# See also: https://docs.ultralytics.com/models/yolo11
import os 
import subprocess # Used to launch tensorboard
import threading # For training to run in a separate thread
import queue # For training progress updates
import signal
import webbrowser # Used to launch tensorboard
import time
import sys
import importlib.util
import shutil
from pathlib import Path
from datetime import datetime
import yaml
from tqdm import tqdm
import numpy as np
from natsort import natsorted
import zarr 
from skimage import measure

import napari
from napari.utils import DirectLabelColormap
# color handling
from octron.sam2_octron.helpers.sam2_colors import (create_label_colors, 
                                                    sample_maximally_different)

from octron.sam2_octron.helpers.video_loader import probe_video

from octron.yolo_octron.helpers.yolo_checks import check_yolo_models
from octron.yolo_octron.helpers.polygons import (find_objects_in_mask, 
                                                 watershed_mask,
                                                 get_polygons,
                                                 polygon_to_mask,
)


from octron.yolo_octron.helpers.yolo_zarr import (create_prediction_store, 
                                                  create_prediction_zarr
)

from octron.yolo_octron.helpers.training import (
    pick_random_frames,
    collect_labels,
    train_test_val,
)
                                                

class YOLO_octron:
    """
    YOLO11 segmentation model class for training with OCTRON data.
    
    This class encapsulates the full pipeline for preparing annotation data from OCTRON,
    generating training datasets, and training YOLO11 models for segmentation tasks.
    It also contains visualization methods for (custom / trained) models.
    
    """
    
    def __init__(self, 
                 models_yaml_path=None,
                 project_path = None,
                 clean_training_dir=True
                 ):
        """
        Initialize YOLO_octron with project and model paths.
        
        Parameters
        ----------
        models_yaml_path : str or Path
            Path to list of available (standard, pre-trained) YOLO models.
        project_path : str or Path, optional
            Path to the OCTRON project directory.
        clean_training_dir : bool
            Whether to clean the training directory if it is not empty.
            Default is True.
            
        """
        self.clean_training_dir = clean_training_dir
        try:
            from ultralytics import settings
            self.yolo_settings = settings
        except ImportError:
            raise ImportError("YOLOv11 is required to run this class.")
        
        # Set up internal variables
        self._project_path = None  # Use private variable for property
        self.training_path = None
        self.data_path = None
        self.model = None
        self.label_dict = None
        self.config_path = None
        self.models_dict = {}
        self.enable_watershed = False
        
        if models_yaml_path is not None:
            self.models_yaml_path = Path(models_yaml_path) 
            if not self.models_yaml_path.exists():
                raise FileNotFoundError(f"Model YAML file not found: {self.models_yaml_path}")

            # Check YOLO models, download if needed
            self.models_dict = check_yolo_models(YOLO_BASE_URL=None,
                                                models_yaml_path=self.models_yaml_path,
                                                force_download=False
                                                )
        else:
            print("No models YAML path provided. Model dictionary will be empty.") 
        
        # If a project path was provided, set it through the property setter
        if project_path is not None:
            self.project_path = project_path  # Uses the property setter
            
            # Setup training directories after project_path is validated
            self._setup_training_directories(self.clean_training_dir)
    
    def __repr__(self):
        """
        Return a string representation of the YOLO_octron object

        """
        pr = f"YOLO_octron(project_path={self.project_path})"
        models = [f"{k}: {v['model_path']}" for k, v in self.models_dict.items()]
        return pr + f"\nModels: {models}"
    
    @property
    def project_path(self):
        """
        Return the project path
        """
        return self._project_path
    
    @project_path.setter
    def project_path(self, path):
        """
        Set the project path with validation
        
        Parameters
        ----------
        path : str or Path
            Path to the OCTRON project directory
            
        Raises
        ------
        FileNotFoundError
            If the path doesn't exist
        TypeError
            If the path is not a string or Path object
        """
        if path is None:
            self._project_path = None
            return
            
        if isinstance(path, str):
            path = Path(path)
        elif not isinstance(path, Path):
            raise TypeError(f"project_path must be a string or Path object, got {type(path)}") 
        # Sanity checks
        if not path.exists():
            raise FileNotFoundError(f"Project path not found: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Project path must be a directory: {path}")
            
        # Path is valid, set it
        self._project_path = path
        print(f"Project path set to: {self._project_path.as_posix()}")
        
        # Update dependent paths if they were previously set
        if self._project_path is not None:
            self.training_path = self._project_path / 'model'
            self.data_path = self.training_path / 'training_data'
            # Setup training directories after project_path is validated
            self._setup_training_directories(self.clean_training_dir)


    def _setup_training_directories(self, clean_training_dir):
        """
        Setup folders for training. 
        This is called from the constructor and when the project path is set.
        
        Parameters
        ----------
        clean_training_dir : bool
            Whether to clean the training directory if it's not empty
        """
        if self._project_path is None:
            raise ValueError("Project path must be set before setting up training directories")
            
        # Setup folders for training
        self.training_path = self._project_path / 'model'  # Path to all training and model output
        self.data_path = self.training_path / 'training_data' # Path to training data
        
        # Folder checks
        try:
            self.training_path.mkdir(exist_ok=False)
        except FileExistsError:
            # Check if training data folder is empty
            if len(list(self.training_path.glob('*'))) > 0:
                if not clean_training_dir:
                    return
                else:
                    shutil.rmtree(self.training_path)
                    self.training_path.mkdir()
                    print(f'Created fresh training directory "{self.training_path.as_posix()}"')       
        return
                    
                    
    ##### TRAINING DATA PREPARATION ###########################################################################    
    def prepare_labels(self, 
                       prune_empty_labels=True, 
                       min_num_frames=5, 
                       verbose=False, 
                       ):
        """ 
        Using collect_labels(), this function finds all object organizer 
        .json files from OCTRON and parses them to extract labels.
        Check collect_labels() function for input arguments.
        """
        
        self.label_dict = collect_labels(self.project_path, 
                                         prune_empty_labels=prune_empty_labels, 
                                         min_num_frames=min_num_frames, 
                                         verbose=verbose
                                        )    
        if verbose: print(f"Found {len(self.label_dict)} organizer files")
    
    
    def prepare_polygons(self):
        """
        Calculate polygons for each mask in each frame and label in the label_dict.
        Optional watershedding is performed on the masks,
        and the polygons are extracted from the resulting labels.
        Whether watershedding is performed is determined by the 
        enable_watershed attribute in this class.

        I am doing some kind of "optimal" watershedding here,
        by determining the median object diameter from a random subset of masks.
        This is then used to determine the min peak distance for the watershedding.
        
        Parameters
        ----------
        label_dict : dict : output from collect_labels()
            Dictionary containing project subfolders,
            and their corresponding labels, annotated frames, masks and video data.
            keys: project_subfolder
            values: dict
                keys: label_id, video
                values: dict, video
                    dict:
                        keys: label, frames, masks, color
                        values: label (str), # Name of the label corresponding to current ID
                                frames (np.array), # Annotated frame indices for the label
                                masks (list of zarr arrays), # Mask zarr arrays
                                color (list) # Color of the label (RGBA, [0,1])
                    video: FastVideoReader object
                    
        Creates
        -------
        label_dict : dict : Dictionary containing project subfolders,
                            and their corresponding labels, annotated frames, masks, polygons and video data.
            keys: project_subfolder
            values: dict
                keys: label_id, video
                values: dict, video
                    dict:
                        keys: label, frames, masks, polygons, color
                        values: label (str), # Name of the label corresponding to current ID
                                frames (np.array), # Annotated frame indices for the label
                                masks (list of zarr arrays), # Mask zarr arrays
                                polygons (dict) # Polygons for each frame index
                                color (list) # Color of the label (RGBA, [0,1])
                    video: FastVideoReader object
        
        Yields
        ------
        no_entry : int
            Number of entry processed
        total_label_dict : int
            Total number of entries in label_dict (all json files)
        label : str
            Current label name
        frame_no : int
            Current frame number being processed
        total_frames : int
            Total number of frames for the current label
            
        
        """ 
        
        # Some constants 
        MIN_SIZE_RATIO_OBJECT_FRAME = 0.00005 # Minimum size ratio of an object to the whole image
                                              # 0.00005: for a 1024x1024 image, this is ~ 52 pixels
        MIN_SIZE_RATIO_OBJECT_MAX = 0.1 # Minimum size ratio of an object to the largest object in the frame
        
        if self.label_dict is None:
            raise ValueError("No labels found. Please run prepare_labels() first.")

        print(f"Watershed: {self.enable_watershed}")
        for no_entry, labels in enumerate(self.label_dict.values(), start=1):  
            min_area = None

            for entry in labels:
                if entry == 'video' or entry == 'video_file_path':
                    continue
                label = labels[entry]['label']
                frames = labels[entry]['frames']
                mask_arrays = labels[entry]['masks'] # zarr array
                
                if self.enable_watershed:
                    # On a subset of masks, determine object properties
                    random_frames = pick_random_frames(frames, n=25)
                    obj_diameters = []
                    for f in random_frames:
                        for mask_array in mask_arrays:
                            sample_mask = mask_array[f]
                            if sample_mask.sum() == 0:
                                continue
                            else:
                                if min_area is None:
                                    # Determine area threshold once
                                    min_area = MIN_SIZE_RATIO_OBJECT_FRAME*sample_mask.shape[0]*sample_mask.shape[1]
                                l, r = find_objects_in_mask(sample_mask, 
                                                           min_area=min_area
                                                           ) 
                                for r_ in r:
                                    # Choosing feret diameter as a measure of object size
                                    # See https://en.wikipedia.org/wiki/Feret_diameter
                                    # and https://scikit-image.org/docs/stable/api/skimage.measure.html
                                    # "Maximum Feret’s diameter computed as the longest distance between 
                                    # points around a region’s convex hull contour
                                    # as determined by find_contours."
                                    obj_diameters.append(r_.feret_diameter_max)
                                    
                    # Now we can make assumptions about the median diameter of the objects
                    # I use this for "optimal" watershed parameters 
                    median_obj_diameter = np.nanmedian(obj_diameters)
                    if np.isnan(median_obj_diameter):
                        median_obj_diameter = 5 # Pretty arbitrary, but should work for most cases
                    if median_obj_diameter < 1:
                        median_obj_diameter = 5
                        
                ##################################################################################
                polys = {} # Collected polygons over frame indices
                for f_no, f in tqdm(enumerate(frames, start=1), 
                            desc=f'Polygons for label {label}', 
                            total=len(frames),
                            unit='frames',
                            leave=True
                            ):    
                    mask_polys = [] # List of polygons for the current frame
                    for mask_array in mask_arrays:
                        mask_current_array = mask_array[f]
                        # Determine area threshold 
                        min_area = MIN_SIZE_RATIO_OBJECT_FRAME*mask_current_array.shape[0]*mask_current_array.shape[1]
                        if self.enable_watershed:
                            # Watershed
                            try:
                                _, water_masks = watershed_mask(mask_current_array,
                                                                footprint_diameter=median_obj_diameter,
                                                                min_size_ratio=MIN_SIZE_RATIO_OBJECT_MAX,  
                                                                plot=False
                                                            )
                            except AssertionError:
                                # The mask is empty at this frame or the object spans the whole frame
                                continue
                            # Loop over watershedded masks
                            for mask in water_masks:
                                try:
                                    mask_polys.append(get_polygons(mask)) 
                                except AssertionError:
                                    # The mask is empty at this frame.
                                    # This happens if there is more than one mask 
                                    # zarr array (because there are multiple instances of a label), 
                                    # and the current label is not present in the current mask array.
                                    pass    
                        else:
                            # No watershedding
                            mask_labeled = np.asarray(measure.label(mask_current_array))
                            unique_labels = np.unique(mask_labeled)
                            assert len(unique_labels) >= 1, f"Labeling failed for {label} in frame {f_no}"
                            # Get new region props to filter out small-ish regions
                            props = measure.regionprops_table(
                                    mask_labeled,
                                    properties=('area','label')
                                    )
                            if not len(props['area']): 
                                continue
                            # Filter out small objects by setting them to 0
                            # and those that are smaller than a certain size ratio 
                            # smaller than the max object size
                            max_area = max(props['area'])
                            for i, area in enumerate(props['area']):
                                if area < min_area:
                                    mask_labeled[mask_labeled == props['label'][i]] = 0          
                                if area < MIN_SIZE_RATIO_OBJECT_MAX*max_area:
                                    mask_labeled[mask_labeled == props['label'][i]] = 0                              
                            if np.sum(mask_labeled) == 0:
                                # No objects found after filtering
                                continue
                            unique_labels = np.unique(mask_labeled)
                            for l in unique_labels:
                                if l == 0:
                                    # Background 
                                    continue
                                else:
                                    # Re-initialize the mask
                                    mask_current_array = np.zeros_like(mask_current_array)
                                    mask_current_array[mask_labeled == l] = 1
                                    mask_polys.append(get_polygons(mask_current_array))

                            
                    polys[f] = mask_polys
                    # Yield, to update the progress bar
                    yield((no_entry, len(self.label_dict), label, f_no, len(frames)))  
                     
                labels[entry]['polygons'] = polys  
            
    
    def prepare_split(self,
                      training_fraction=0.7,
                      validation_fraction=0.15,
                      verbose=False,
                     ):
        """
        Using train_test_val(), this function splits the frame indices 
        into training, testing, and validation sets, based on the fractions provided.
        """
        if self.label_dict is None:
            raise ValueError("No labels found. Please run prepare_labels() first.")
        
        for labels in self.label_dict.values():
            for entry in labels:
                if entry == 'video' or entry == 'video_file_path':
                    continue    
                # label = labels[entry]['label']
                frames = labels[entry]['frames']   
                split_dict = train_test_val(frames, 
                                            training_fraction=training_fraction,
                                            validation_fraction=validation_fraction,
                                            verbose=verbose,
                                            )

                labels[entry]['frames_split'] = split_dict
        
    
    def create_training_data(self,
                             verbose=False,
                            ):
        """
        Create training data for YOLO segmentation.
        This function exports the training data to the data_path folder.
        The training data consists of images and corresponding label text files.
        The label text files contain the label ID and normalized polygon coordinates.
        
        Parameters
        ----------
        verbose : bool
            Whether to print progress messages

        Yields
        ------
        no_entry : int
            Number of entry processed
        total_label_dict : int
            Total number of entries in label_dict (all json files)
        label : str
            Current label name
        split : str
            Current split (train, val, test)
        frame_no : int
            Current frame number being processed
        total_frames : int
            Total number of frames for the current label
                     
        
        """
        if self.data_path is None:
            raise ValueError("No data path set. Please set 'project_path' first.")
        if self.training_path is None:
            raise ValueError("No training path set. Please set 'project_path' first.")
        if self.label_dict is None:
            raise ValueError("No labels found. Please run prepare_labels() first.")
        
        try:
            from PIL import Image
        except ModuleNotFoundError:
            print('Please install PIL first, via pip install pillow')
            return
        
        # Completeness checks
        for labels in self.label_dict.values(): 
            for entry in labels:
                if entry == 'video' or entry == 'video_file_path':
                    continue
                assert 'frames' in labels[entry], "No frame indices (frames) found in labels"
                assert 'polygons' in labels[entry], "No polygons found in labels, run prepare_polygons() first"
                assert 'frames_split' in labels[entry], "No data split found in labels, run prepare_split() first"  

        # Create the training root directory
        # If it already exists, delete it and create a new one
        "self.training_path"
        if self.data_path.exists() and self.clean_training_dir:
            raise FileExistsError(
                f"Training data path '{self.data_path.as_posix()}' already exists. "
                "Please remove it or set self.clean_training_dir=False."
            )
        if self.data_path.exists() and not self.clean_training_dir:
            print(f"Training data path '{self.data_path.as_posix()}' already exists. Using existing directory.")
            # Remove any model subdirectories
            if self.training_path / 'training' in self.training_path.glob('*'):
                shutil.rmtree(self.training_path / 'training')
                print(f"Removed existing model subdirectory '{self.training_path / 'training'}'")
            return
        if not self.data_path.exists():
            self.data_path.mkdir(parents=True, exist_ok=False)
            print(f"Created training data directory '{self.data_path.as_posix()}'")
            

        # Create subdirectories for train, val, and test
        # If they already exist, delete them and create new ones
        for split in ['train', 'val', 'test']:
            path_to_split = self.data_path / split
            try:
                path_to_split.mkdir(exist_ok=False)
            except FileExistsError:
                shutil.rmtree(path_to_split)    
                path_to_split.mkdir()

        #######################################################################################################
        # Export the training data
        
        for no_entry, (path, labels) in enumerate(self.label_dict.items(), start=1):  
            path_prefix = Path(path).name   
            video_data = labels.pop('video')
            _ = labels.pop('video_file_path')
            for entry in tqdm(labels,
                            total=len(labels),
                            position=0,
                            unit='labels',
                            leave=True,
                            desc=f'Exporting {len(labels)} label(s)'
                            ):
                current_label_id = entry
                label = labels[entry]['label']  
                # Extract the size of the masks for normalization later on 
                for m in labels[entry]['masks']:
                    assert m.shape == labels[entry]['masks'][0].shape, f'All masks should have the same shape'
                _, mask_height, mask_width = labels[entry]['masks'][0].shape
                
                for split in ['train', 'val', 'test']:
                    current_indices = labels[entry]['frames_split'][split]
                    for frame_no, frame_id in tqdm(enumerate(current_indices),
                                                    total=len(current_indices), 
                                                    desc=f'Exporting {split} frames', 
                                                    position=1,    
                                                    unit='frames',
                                                    leave=False,
                                                    ):
                        frame = video_data[frame_id]
                        image_output_path = self.data_path / split / f'{path_prefix}_{frame_id}.png'
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
                                compress_level=0, # 0-9, lower means higher quality
                                optimize=True,
                            )
                        
                        # Create the label text file with the correct format
                        with open(self.data_path / split / f'{path_prefix}_{frame_id}.txt', 'a') as f:
                            for polygon in labels[entry]['polygons'][frame_id]:
                                f.write(f'{current_label_id}')
                                # Write each coordinate pair as normalized coordinate to txt
                                for point in polygon:
                                    f.write(f' {point[0]/mask_width} {point[1]/mask_height}')
                                f.write('\n')
                                
                        # Yield, to update the progress bar
                        yield((no_entry, len(self.label_dict), label, split, frame_no, len(current_indices)))  
                        
        if verbose: print(f"Training data exported to {self.data_path.as_posix()}")
        return

    def write_yolo_config(self,
                         train_path="train",
                         val_path="val",
                         test_path="test",
                        ):
        """
        Write the YOLO configuration file for training.
        
        Parameters
        ----------
        train_path : str
            Path to training data (subfolder of self.data_path)
        val_path : str
            Path to validation data (subfolder of self.data_path)
        test_path : str
            Path to test data (subfolder of self.data_path)
            
        """
        if self.label_dict is None:
            raise ValueError("No labels found.")
        
        dataset_path = self.data_path
        assert dataset_path is not None, f"Data path not set. Please set 'project_path' first."
        assert dataset_path.exists(), f'Dataset path not found at {dataset_path}'
        assert dataset_path.is_dir(), f'Dataset path should be a directory, but found a file at {dataset_path}' 
        
        if len(list(dataset_path.glob('*'))) <= 1:
            raise FileNotFoundError(
                f"No training data found in {dataset_path.as_posix()}. Please run create_training_data() first."
                )
        if (not (dataset_path / "train").exists() 
            or not (dataset_path / "val").exists() 
            or not (dataset_path / "test").exists()
            ):
            raise FileNotFoundError(
                f"Training data not found(train/val/test). Please run create_training_data() first."
                )   
        
        # Get label names from the object organizer
        label_id_label_dict = {}
        for labels in self.label_dict.values():
            for entry in labels:
                if entry == 'video' or entry == 'video_file_path':
                    continue   
                if entry in label_id_label_dict:
                    assert label_id_label_dict[entry] == labels[entry]['label'],\
                        f"Label mismatch for {entry}: {label_id_label_dict[entry]} vs {labels[entry]['label']}"
                else:
                    label_id_label_dict[entry] = labels[entry]['label']

        ######## Write the YAML config
        self.config_path = dataset_path / "yolo_config.yaml"

        # Create the config dictionary
        config = {
            "path": str(dataset_path),
            "train": train_path,
            "test": test_path,
            "val": val_path,
            "names": label_id_label_dict,
        }
        header = "# OCTRON training config\n# Last edited on {}\n\n".format(datetime.now())
        
        # Write to file
        with open(self.config_path, 'w') as f:
            f.write(header)
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"YOLO config saved to '{self.config_path.as_posix()}'")


    ##### TRAINING AND INFERENCE ############################################################################
    def load_model(self, model_name_path):
        """
        Load the YOLO model
        
        Parameters
        ----------
        model_name_path : str or Path
            Path to the model to load, or name of the model to load
            (e.g. 'YOLO11m-seg'), defaults to the model in the models.yaml file.
        
        Returns
        -------
        model : YOLO
            Loaded YOLO model
        """
         
        # Configure YOLO settings
        if not hasattr(self, 'training_path') or self.training_path is None:
            pass
        else:
            self.yolo_settings.update({
                'sync': False,
                'hub': False,
                'tensorboard': True,
                'runs_dir': self.training_path.as_posix()
            })
        from ultralytics import YOLO   
        
        # Load specified model
        try:
            assert Path(model_name_path).exists()
            # If this path exists, load this model, otherwise 
            # assume that this models is part of the models_dict
        except AssertionError:
            model_name_path = self.models_dict[model_name_path]['model_path']
            model_name_path = self.models_yaml_path.parent / f'models/{model_name_path}'    
            
        model = YOLO(model_name_path)
        print(f"Model loaded from '{model_name_path.as_posix()}'")
        self.model = model
        return model
    
    def load_model_args(self, model_name_path):
        """
        Load the YOLO model args.yaml (model training settings).
        This file is supposed to be one level up of the "weights" folder for 
        custom trained models.
        
        Parameters
        ----------
        model_name_path : str or Path
            Path to the model to load, or name of the model to load
        
        Returns
        -------
        args : dict
            Dictionary containing the model arguments
            Returns None if the args.yaml file is not found
            
        """
        model_name_path = Path(model_name_path)
                
        assert model_name_path.exists(), f"Model path {model_name_path} does not exist."
        model_parent_path = model_name_path.parent.parent
        args = list(model_parent_path.glob('args.yaml'))
        if len(args) > 0: 
            args = args[0]  
            # Read yaml as dict
            with open(args, 'r') as f:
                args = yaml.safe_load(f)
        else: 
            args = None
        
        return args
    

    def train(self, 
              device='cpu',
              imagesz = 640,    
              epochs=30, 
              save_period=15,
              ):
        """
        Train the YOLO model with epoch progress updates
        
        Parameters
        ----------
        device : str
            Device to use ('cpu', 'cuda', 'mps')
        imagesz : int
            Input image size
        epochs : int
            Number of epochs to train for
        save_period : int
            Save model every n epochs
            
        Yields
        ------
        dict
            Progress information including:
            - epoch: Current epoch
            - total_epochs: Total number of epochs
            - epoch_time: Time taken for current epoch
            - estimated_finish_time: Estimated finish time
        """
        if not self.model:
            print('😵 No model loaded!')
            return
            
        # Clear any existing callbacks
        if hasattr(self.model, 'callbacks'):
            for callback_name in ['on_fit_epoch_end', 'on_train_start', 'on_train_end']:
                if callback_name in self.model.callbacks:
                    self.model.callbacks.pop(callback_name, None)
                    
        # Setup a queue to receive yielded values from the callback
        progress_queue = queue.Queue()
        
        # Track last epoch seen to avoid duplicates
        last_epoch_reported = -1
        # Internal callback to capture training progress
        def _on_fit_epoch_end(trainer):
            nonlocal last_epoch_reported
            current_epoch = trainer.epoch + 1
            
            # Skip if we already reported this epoch (prevents duplicates)
            if current_epoch <= last_epoch_reported:
                return
                
            last_epoch_reported = current_epoch
            
            # Calculate progress information
            epoch_time = trainer.epoch_time
            remaining_time = epoch_time * (epochs - current_epoch)
            finish_time = time.time() + remaining_time
            
            # Put the information in the queue
            progress_queue.put({
                'epoch': current_epoch,
                'total_epochs': epochs,
                'epoch_time': epoch_time,
                'remaining_time': remaining_time,
                'finish_time': finish_time,
            })
        
        # Add our callback that will put progress info into the queue
        self.model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
        
        if self.config_path is None or not self.config_path.exists():
            raise FileNotFoundError(
                "No configuration .yaml file found."
            )
        if device not in ['cpu', 'cuda', 'mps']:
            raise ValueError(f"Invalid device: {device}")   
        if device == 'mps':
            print("⚠ MPS is not yet fully supported in PyTorch. Use at your own risk.")
        
        assert imagesz % 32 == 0, 'YOLO image size must be a multiple of 32'

        # Start training in a separate thread
        training_complete = threading.Event()
        training_error = None
        
        def run_training():
            # https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings
            # overlap_mask - https://github.com/ultralytics/ultralytics/issues/3213#issuecomment-2799841153
            nonlocal training_error
            try:
                # Start training
                print(f"Starting training for {epochs} epochs...")
                self.model.train(
                    data=self.config_path.as_posix(), 
                    name='training',
                    project=self.training_path.as_posix(),
                    mode='segment',
                    device=device,
                    optimizer='auto',
                    rect=False,
                    cos_lr=True,
                    mask_ratio=2,
                    overlap_mask=True,
                    fraction=1.0,
                    epochs=epochs,
                    imgsz=imagesz,
                    resume=False,
                    patience=50,
                    plots=True,
                    batch=-1,
                    cache='disk',
                    save=True,
                    save_period=save_period, 
                    exist_ok=True,
                    # Augmentation
                    hsv_v=.25,
                    hsv_s=0.0,
                    hsv_h=0.0,
                    degrees=180,
                    translate=0.1,
                    perspective=0,
                    scale=0.25,
                    shear=2,
                    flipud=.5,
                    fliplr=.5,
                    mosaic=0.25,
                    mixup=0.25,
                    copy_paste=0.25,
                    copy_paste_mode='mixup', 
                    erasing=0.,
                )
            except Exception as e:
                training_error = e
            finally:
                # Signal that training is complete
                training_complete.set()
        
        # Start training thread
        training_thread = threading.Thread(target=run_training)
        training_thread.daemon = True
        training_thread.start()
        
        try:
            # Monitor progress queue and yield updates until training completes
            while not training_complete.is_set() or not progress_queue.empty():
                try:
                    # Wait for progress info with timeout
                    progress_info = progress_queue.get(timeout=1.0)
                    yield progress_info
                    progress_queue.task_done()
                except queue.Empty:
                    # No progress info available yet, continue waiting
                    pass
                    
            # If there was an error in the training thread, raise it
            if training_error:
                raise training_error
                
        except KeyboardInterrupt:
            print("Training interrupted by user")
            
    
    def launch_tensorboard(self):
        """
        Check if TensorBoard is installed, launch it with the training directory,
        and open a web browser to view the TensorBoard interface.
        Chooses a random port every time to avoid port collisions.
        If TensorBoard is not installed, it will attempt to install it using pip.
        
        
        Parameters
        ----------

        Returns
        -------
        bool
            True if TensorBoard was successfully launched, False otherwise
        """
        import random
        # Check if tensorboard is installed
        if importlib.util.find_spec("tensorboard") is None:
            print("TensorBoard is not installed. Installing now...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboard"])
                print("TensorBoard installed successfully!")
            except subprocess.CalledProcessError:
                print("Failed to install TensorBoard. Please install it manually with:")
                print("pip install tensorboard")
                return False
        
        if self.training_path is None:
            print("No training path set. Set project_path first.")
            return False
            
        if not self.training_path.exists():
            print(f"Training path '{self.training_path}' does not exist.")
            return False
        
        # Launch tensorboard in a separate process
        log_dir = self.training_path / 'training'
        try:
            port = random.randint(6000, 7000)
            print(f"Starting TensorBoard on port {port}...")
            tensorboard_process = subprocess.Popen(
                [sys.executable, "-m", "tensorboard.main", 
                "--logdir", log_dir.as_posix(),
                "--port", str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Give it a moment to start up
            time.sleep(3)
            
            # Check if process is still running
            if tensorboard_process.poll() is not None:
                # Process terminated - get error message
                _, stderr = tensorboard_process.communicate()
                print(f"Failed to start TensorBoard: {stderr}")
                return False
                
            # Open web browser
            tensorboard_url = f"http://localhost:{port}/"
            print(f"Opening TensorBoard in browser: {tensorboard_url}")
            webbrowser.open(tensorboard_url)
            
            print("TensorBoard is running.")
            return True
            
        except Exception as e:
            print(f"Error launching TensorBoard: {e}")
            return False
        

    def _quit_tensorboard_posix(self):
        """
        Helper method to terminate TensorBoard on Unix-like systems
        """
        # Find processes with tensorboard in the command
        result = subprocess.run(
            ["ps", "-ef"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        lines = result.stdout.split('\n')
        found_processes = False
        
        for line in lines:
            if 'tensorboard.main' in line or 'tensorboard ' in line:
                # Extract PID and kill
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        pid = int(parts[1])
                        print(f"Terminating TensorBoard process with PID {pid}")
                        os.kill(pid, signal.SIGTERM)
                        found_processes = True
                    except (ValueError, ProcessLookupError) as e:
                        print(f"Failed to terminate TensorBoard process: {e}")
        
        if not found_processes:
            print("No TensorBoard processes found")

    def _quit_tensorboard_windows(self):
        """Helper method to terminate TensorBoard on Windows"""
        # Use tasklist and taskkill on Windows
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"], 
            capture_output=True, 
            text=True
        )
        found_processes = False
        
        for line in result.stdout.split('\n'):
            if 'tensorboard' in line.lower():
                try:
                    parts = line.strip('"').split('","')
                    if len(parts) >= 2:
                        pid = int(parts[1])
                        print(f"Terminating TensorBoard process with PID {pid}")
                        subprocess.run(["taskkill", "/F", "/PID", str(pid)])
                        found_processes = True
                except (ValueError, IndexError) as e:
                    print(f"Failed to terminate TensorBoard process: {e}")
        
        if not found_processes:
            print("No TensorBoard processes found")

    
    def quit_tensorboard(self):
        """
        Find and quit all TensorBoard processes on both Unix-like systems 
        and Windows platforms.
        """
        print("Stopping any running TensorBoard processes...")
        
        try:
            # Check platform-specific approach
            if os.name == 'posix':  # Unix-like systems (macOS, Linux)
                self._quit_tensorboard_posix()
            elif os.name == 'nt':  # Windows
                self._quit_tensorboard_windows()
            else:
                print(f"Unsupported platform: {os.name}")
        except Exception as e:
            print(f"Error when terminating TensorBoard processes: {e}")
    
     
    def validate(self, data=None, device='auto', plots=True):
        """
        Validate the model
        
        Parameters
        ----------
        data : str or Path, optional
            Path to validation data, defaults to the validation set in the config
        device : str
            Device to use for inference
        plots : bool
            Whether to generate plots
            
        Returns
        -------
        metrics : dict
            Validation metrics
        """
        # TODO: Which model to validate
        # Should be able to choose checkpoint, like best, last, etc.
        
        # if self.model is None:
        #     self.load_model()
            
        # data_path = data if data else self.config_path
        # print(f"Running validation on {data_path}...")
        
        # metrics = self.model.val(data=data_path, device=device, plots=plots)
        
        # print("Validation results:")
        # print(f"Mean Average Precision for boxes: {metrics.box.map}")
        # print(f"Mean Average Precision for masks: {metrics.seg.map}")
        
        # return metrics
        pass
    
    def find_trained_models(self, 
                            project_path, 
                            subfolder_route='training/weights'
                            ):
        """
        Find all trained models in the training directory
        
        Parameters
        ----------
        project_path : str or Path
            Path to the project directory
        subfolder_route : str
            Subfolder route to the models. 
            This defaults to 'training/weights' for trained OCTRON YOLO models.   
            
        """
        project_path = Path(project_path)
        assert project_path.exists(), f"Project path not found: {project_path}"
        assert project_path.is_dir(), f"Project path must be a directory: {project_path}"
        found_models_project = natsorted(project_path.rglob(f'*{subfolder_route}/*.pt'))
        return found_models_project 
    
    def write_bot_sort_yaml(self, output_path):
        """
        Write a BotSort tracker configuration to a YAML file
        
        """
        
        assert output_path.suffix == '.yaml', "Tracker yaml path must have a .yaml extension" 
        output_path = Path(output_path)
        assert output_path.parent.exists(), "Tracker output path parent directory does not exist"
        
        # Define the tracker configuration
        tracker_config = {
            'tracker_type': 'botsort',    # tracker type, ['botsort', 'bytetrack']
            'track_high_thresh': 0.6,     # threshold for the first association
            'track_low_thresh': 0.2,      # threshold for the second association
            'new_track_thresh': 0.8,      # threshold for init new track if the detection does not match any tracks
            'track_buffer': 100_000,      # buffer to calculate the time when to remove tracks
            'match_thresh': 0.95,         # threshold for matching tracks
            'fuse_score': False,          # Whether to fuse confidence scores with the iou distances before matching
            # 'min_box_area': 10,         # threshold for min box areas(for tracker evaluation, not used for now)
            # BoT-SORT settings
            'gmc_method' : 'sparseOptFlow',
            'downscale'  : 3,
            # ReID model related thresh (not supported yet)
            'proximity_thresh'  : 0.5,
            'appearance_thresh' : 0.5,
            'with_reid' : False,
        }

        # Add a header comment to the YAML file
        header = """# Tracker configuration for BoT-SORT\n# For BoT-SORT source code see https://github.com/NirAharon/BoT-SORT\n"""

        # Write the configuration to a YAML file with header
        with open(output_path, 'w') as f:
            f.write(header)
            yaml.dump(tracker_config, f, default_flow_style=False, sort_keys=False)

        print(f"BoT-SORT configuration written to {output_path.absolute()}")
      
      
    
    def write_byte_tracker_yaml(self, output_path):
        """
        Write a ByteTrack tracker configuration to a YAML file
        
        """
        
        assert output_path.suffix == '.yaml', "Tracker yaml path must have a .yaml extension" 
        output_path = Path(output_path)
        assert output_path.parent.exists(), "Tracker output path parent directory does not exist"
        
        # Define the tracker configuration
        tracker_config = {
            'tracker_type': 'bytetrack',  # tracker type, ['botsort', 'bytetrack']
            'track_high_thresh': 0.6,     # threshold for the first association
            'track_low_thresh': 0.2,      # threshold for the second association
            'new_track_thresh': 0.8,      # threshold for init new track if the detection does not match any tracks
            'track_buffer': 100_000,      # buffer to calculate the time when to remove tracks
            'match_thresh': 0.95,         # threshold for matching tracks
            'fuse_score': False,          # Whether to fuse confidence scores with the iou distances before matching
            # 'min_box_area' : 100,       # (Seems like this parameter is not used )
        }


        # Add a header comment to the YAML file
        header = """# Tracker configuration for ByteTrack\n# For ByteTrack source code see https://github.com/ifzhang/ByteTrack\n"""

        # Write the configuration to a YAML file with header
        with open(output_path, 'w') as f:
            f.write(header)
            yaml.dump(tracker_config, f, default_flow_style=False, sort_keys=False)

        print(f"ByteTrack configuration written to {output_path.absolute()}")
      
      
    
    def predict_batch(self, 
                      videos_dict,
                      model_path,
                      device,
                      tracker_name,
                      one_object_per_label,
                      iou_thresh=.9,
                      conf_thresh=.8,
                      polygon_sigma=1.,
                      overwrite=True
                      ):
        """
        Predict and track objects in multiple videos.
        
        Parameters
        ----------
        videos_dict : dict
            Dictionary of video paths and video dictionaries with metadata.
        model_path : str or Path
            Path to the YOLO model to use for prediction.
        device : str
            Device to run prediction on ('cpu', 'cuda', etc.)
        tracker_name : str
            Name of the tracker to use ('bytetrack' or 'botsort')
        one_object_per_label : bool
            Whether to track only one object per label.
            If True, only the first detected object of each label will be tracked
            and if more than one object is detected, only the first one with the highest confidence
            will be kept.
        iou_thresh : float
            IOU threshold for detection
        conf_thresh : float
            Confidence threshold for detection
        polygon_sigma : float
            Sigma value for polygon smoothing
        overwrite : bool
            Whether to overwrite existing prediction results
            
        Yields
        ------
        dict
            Progress information including:
            - stage: Current stage ('initializing', 'processing')
            - video_name: Current video name
            - video_index: Index of current video
            - total_videos: Total number of videos
            - frame: Current frame
            - total_frames: Total frames in current video
            - fps: Processing speed (frames per second)
            - eta: Estimated time remaining in seconds
            - eta_finish_time: Estimated finish time timestamp
            - overall_progress: Overall progress as percentage (0-100)
        """

        available_trackers = ['bytetrack','botsort']
        tracker_name = tracker_name.strip().lower()
        assert tracker_name in available_trackers, f'Tracker with name {tracker_name} not available.'
        
        model_path = Path(model_path)
        assert model_path.exists(), f"Model path {model_path} does not exist."
        # Try to find model args 
        model_args = self.load_model_args(model_name_path=model_path)
        if model_args is not None:
            print('Model args loaded from', model_path.parent.parent.as_posix())
            imgsz = model_args['imgsz']
            print(f'Image size: {imgsz}')
        else:
            print('No model args found, using default image size of 640')
            imgsz = 640
        
        if one_object_per_label:
            print("⚠ Tracking only one object per label.")
        
        # Calculate total frames across all videos
        total_videos = len(videos_dict)
        total_frames = sum(v['num_frames'] for v in videos_dict.values())
        
        # Process each video
        for video_index, (video_name, video_dict) in enumerate(videos_dict.items(), start=0):
            # Load model anew for every video since the tracker persists
            try:
                model = self.load_model(model_name_path=model_path)
                if not model:
                    print(f"Failed to load model from {model_path}")
                    return
            except Exception as e:
                print(f"Error during initialization: {e}")
                return    

            print(f'\nProcessing video {video_index+1}/{total_videos}: {video_name}')
            video_path = Path(video_dict['video_file_path'])
            num_frames = video_dict['num_frames']
            
            if max(video_dict['height'], video_dict['width']) < imgsz:
                print(f"⚠ Video resolution is smaller than the model image size ({imgsz}). Setting retina_masks to False.")
                retina_masks = False
            else:
                retina_masks = True
            
            # Set up prediction directory structure
            save_dir = video_path.parent / 'predictions' / f"{video_path.stem}_{tracker_name}"
            if save_dir.exists() and overwrite:
                shutil.rmtree(save_dir)
            elif save_dir.exists() and not overwrite:
                print(f"Prediction directory already exists at {save_dir}")
                continue
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Set up tracker 
            tracker_yaml_path = save_dir / f'{tracker_name}.yaml'
            if not tracker_yaml_path.exists():
                if tracker_name == 'bytetrack':
                    self.write_byte_tracker_yaml(tracker_yaml_path)
                elif tracker_name == 'botsort':
                    self.write_bot_sort_yaml(tracker_yaml_path)
                    
            # Prepare prediction stores
            prediction_store_dir = save_dir / 'predictions.zarr'
            prediction_store = create_prediction_store(prediction_store_dir)
            zarr_root = zarr.open_group(store=prediction_store, mode='a')
            
            # Process video frames
            video = video_dict['video']
            tracking_df_dict = {}
            track_id_label_dict = {} # Keeps track of label - ID pairs, depending on one_object_per_label
            frame_start = time.time()
            for frame_no, frame in enumerate(video, start=0):
                # Before processing the results, yield progress information 
                # This is because we want this information regardless of whether there 
                # were any detections in the frame
                # Update timing information
                frame_time = time.time()-frame_start
                yield {
                    'stage': 'processing',
                    'video_name': video_name,
                    'video_index': video_index,
                    'total_videos': total_videos,
                    'frame': frame_no + 1,
                    'total_frames': num_frames,
                    'frame_time': frame_time,
                }
                frame_start = time.time()
                
                # Run tracking on this frame
                results = model.track(
                    source=frame, 
                    persist=True,
                    task='segment',
                    tracker=tracker_yaml_path.as_posix(),
                    project=save_dir.parent.as_posix(),
                    name=save_dir.name,
                    show=False,
                    save=False,
                    verbose=False,
                    imgsz=imgsz,
                    stream_buffer=True,
                    conf=conf_thresh,
                    iou=iou_thresh,
                    device=device, 
                    retina_masks=retina_masks, # original image resolution, not inference resolution?
                    show_boxes=False,
                    save_txt=False,
                    save_conf=False,
                )

                # Process results and save to zarr/CSV here
                confidences = results[0].boxes.conf.cpu().numpy()
                label_names = tuple([results[0].names[int(r)] for r in results[0].boxes.cls.cpu().numpy()])
                
                # Then process the results ...    
                try:
                    masks_polys = results[0].masks.xy  
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                except AttributeError:
                    # Empty prediction
                    continue
                
                # Extract tracks 
                for label, track_id, conf, mask_poly in zip(label_names, 
                                                            track_ids,
                                                            confidences,
                                                            masks_polys, 
                                                            ):
                    
                    if one_object_per_label or iou_thresh < 0.01:
                        # ! Use 'label' as keys in track_id_label_dict
                        # There is only one object/track ID per label
                        if label in track_id_label_dict:
                            # Overwrite whatever current track ID is assigned to this label
                            track_id = track_id_label_dict[label]
                        else:
                            # Assign a new, custom track ID
                            track_id = max(track_id_label_dict.values(), default=0) + 1
                            track_id_label_dict[label] = track_id
                    else: 
                        # ! Use 'track_id' as keys in track_id_label_dict
                        # There can be multiple objects/track IDs per label
                        if track_id in track_id_label_dict:
                            label_ = track_id_label_dict[track_id]
                            if label_ != label: 
                                # This happens in rare cases 
                                # if the same track ID is assigned to different labels over time 
                                # Assign a new track ID
                                track_id = max(track_id_label_dict.keys(), default=0) + 1
                                track_id_label_dict[track_id] = label
                        else:
                            track_id_label_dict[track_id] = label   
                        
                    # Initialize mask zarr array if needed
                    if f'{track_id}_masks' not in list(zarr_root.array_keys()):
                        video_shape = (video_dict['num_frames'], video_dict['height'], video_dict['width'])   
                        mask_store = create_prediction_zarr(prediction_store, 
                                        f'{track_id}_masks',
                                        shape=video_shape,
                                        chunk_size=50,     
                                        fill_value=-1,
                                        dtype='int8',                           
                                        video_hash=''
                                        )
                        mask_store.attrs['label'] = label
                        mask_store.attrs['classes'] = results[0].names
                    else:
                        mask_store = zarr_root[f'{track_id}_masks']
                        
                    # Initialize tracking dataframe if needed
                    if track_id not in tracking_df_dict:
                        tracking_df = self.create_tracking_dataframe(video_dict)
                        tracking_df.attrs['video_name'] = video_name
                        tracking_df.attrs['label'] = label
                        tracking_df.attrs['track_id'] = track_id
                        tracking_df_dict[track_id] = tracking_df
                    else:
                        tracking_df = tracking_df_dict[track_id]
                        assert tracking_df.attrs['track_id'] == track_id, "ID mismatch" 
                        assert tracking_df.attrs['label'] == label, "Label mismatch"    

                    # Check if a row already exists and compare current confidence with existing one
                    # This happens if one_object_per_label is True or iou_thresh < 0.01 
                    # and there are multiple detections
                    if (frame_no, track_id) in tracking_df.index:
                        existing_conf = tracking_df.loc[(frame_no, track_id), 'confidence']
                        if conf <= existing_conf and iou_thresh >= 0.01:
                            # Skip this detection if a better one already exists
                            # and we are not fusing masks (iou_thresh > 0)
                            continue
                        else:
                            # Average the confidence values
                            conf = (conf + existing_conf) / 2
                        
                    # Continue to create masks and extract properties
                    dummy_mask = np.zeros((video_dict['height'], video_dict['width']))    
                    mask = polygon_to_mask(dummy_mask.astype('int8'), 
                                           mask_poly, 
                                           smooth_sigma=polygon_sigma,
                                           opening_radius=2,
                                           model_imgsz=imgsz,
                                           )
                    if iou_thresh < 0.01:
                        # Fuse this masks with prior masks already stored in the zarr
                        # for this frame / label
                        previous_mask = mask_store[frame_no,:,:].copy()
                        previous_mask[previous_mask == -1] = 0
                        mask = np.logical_or(previous_mask, mask)
                        mask = mask.astype('int8')
                    
                    mask_store[frame_no,:,:] = mask
                        
                    # Get region properties and save them to the dataframe
                    _, regions_props = find_objects_in_mask(mask, min_area=0)
                    # Instead of asserting single region, handle multiple regions
                    if not regions_props:
                        # Skip if no regions were found
                        continue
                    num_regions = len(regions_props)                    
                    # Initialize accumulators for region properties
                    pos_x_sum, pos_y_sum = 0, 0
                    area_sum = 0
                    eccentricity_sum = 0
                    orientation_sum = 0
                    solidity_sum = 0
                    # Loop over all regions and accumulate properties
                    for region_prop in regions_props:
                        centroid = region_prop.centroid
                        pos_x_sum += centroid[1]  # x coordinate
                        pos_y_sum += centroid[0]  # y coordinate
                        area_sum += region_prop.area
                        eccentricity_sum += region_prop.eccentricity
                        orientation_sum += region_prop.orientation
                        solidity_sum += region_prop.solidity
                    
                    # Store averages in DataFrame with flat column names
                    tracking_df.loc[(frame_no, track_id), 'pos_x'] = pos_x_sum / num_regions
                    tracking_df.loc[(frame_no, track_id), 'pos_y'] = pos_y_sum / num_regions
                    tracking_df.loc[(frame_no, track_id), 'area'] = area_sum / num_regions
                    tracking_df.loc[(frame_no, track_id), 'eccentricity'] = eccentricity_sum / num_regions
                    tracking_df.loc[(frame_no, track_id), 'orientation'] = orientation_sum / num_regions
                    tracking_df.loc[(frame_no, track_id), 'confidence'] = conf
                    tracking_df.loc[(frame_no, track_id), 'solidity'] = solidity_sum / num_regions

                # A FRAME IS COMPLETE
                
            # A VIDEO IS COMPLETE 
            # Save each tracking DataFrame with a label column added
            for track_id, tr_df in tracking_df_dict.items():
                label = tr_df.attrs["label"]
                df_to_save = tr_df.copy()
                # Add the label column (will be filled with the same value for all rows)
                df_to_save.insert(0, 'label', label)
                
                # Save to CSV with metadata header
                filename = f'{label}_track_{track_id}.csv'
                csv_path = save_dir / filename
                
                # Create header with metadata
                header = [
                    f"video_name: {tr_df.attrs.get('video_name', 'unknown')}",
                    f"frame_count: {tr_df.attrs.get('frame_count', '')}",
                    f"video_height: {tr_df.attrs.get('video_height', '')}",
                    f"video_width: {tr_df.attrs.get('video_width', '')}",
                    f"created_at: {tr_df.attrs.get('created_at', str(datetime.now()))}",
                    ""  # Empty line to separate header from data
                ]
                
                # Write the header and then the data
                with open(csv_path, 'w') as f:
                    f.write('\n'.join(header))
                    df_to_save.to_csv(f)
                print(f"Saved tracking data for '{label}' (track ID: {track_id}) to {filename}")
            
            yield {
                    'stage': 'video_complete',
                    'save_dir': save_dir,
                }
            
        # ALL COMPLETE    
        yield {
            'stage': 'complete',
            'total_videos': total_videos,
            'total_frames': total_frames,
        }
                
    
    def create_tracking_dataframe(self, video_dict):
        """
        Create an empty DataFrame for storing tracking data and associated metadata
        I am using the video_dict to get number of frames that are expected for the 
        tracking dataframe and to store the video metadata in the DataFrame attributes.
        
        Parameters
        ----------
        video_dict : dict
            Dictionary with video metadata including num_frames

            
        Returns
        -------
        pd.DataFrame
            Empty DataFrame initialized for tracking data
        """
        import pandas as pd
        assert 'num_frames' in video_dict, "Video metadata must include 'num_frames'"
        # Create a flat column structure
        columns = ['confidence', 'pos_x', 'pos_y', 'area', 'eccentricity', 'orientation','solidity']
        
        # Initialize the DataFrame with NaN values
        df = pd.DataFrame(
            index=pd.MultiIndex.from_product([
                range(video_dict['num_frames']),  # Frame numbers
                []  # Empty track_id list - will be populated during tracking
            ], names=['frame', 'track_id']),
            columns=columns,

        )
        
        # Add metadata as DataFrame attributes
        df.attrs = {
            'video_hash': video_dict.get('hash', ''), 
            'video_name': None,  # Will be filled in later
            'video_height': video_dict.get('height', np.nan),
            'video_width': video_dict.get('width', np.nan),
            'frame_count': video_dict['num_frames'],
            'created_at': str(datetime.now())
        }
        
        return df

    def show_predictions(self, 
                         save_dir,
                         sigma_tracking_pos=2
                         ):
        """
        Show the predictions in the current save directory in a 
        new napari viewer.
        
        Parameters
        ----------
        save_dir : str or Path  
            Path to the directory with the predictions
        sigma_tracking_pos : int
            Sigma value for tracking position smoothing
            CURRENTLY FIXED TO 1
            
        Yields
        -------
        label : str
            Label of the tracked object
        track_id : int
            Track ID of the tracked object
        track_df_napari : pd.DataFrame
            DataFrame with tracking data for the object (xy coordinates)
        features_df_napari: pd.DataFrame
            DataFrame with features for the object (area, eccentricity, etc.)
                    
            
        """
        # Some heavy imports are hidden here
        from scipy.ndimage import gaussian_filter1d
        import pandas as pd 
        # Napari PyAV reader 
        from napari_pyav._reader import FastVideoReader
        

        
        def _find_class_by_name(classes, class_name):
            return (next((k for k, v in classes.items() if v == class_name), None))
        
        save_dir = Path(save_dir)
        if not save_dir.exists():
            print(f"Save directory not found: {save_dir}")
            return
        # Check if video is present in the parent directory
        video = None
        for video_path in save_dir.parent.parent.glob('*.mp4'):
            if video_path.stem == '_'.join(save_dir.name.split('_')[:-1]):
                video = FastVideoReader(video_path, read_format='rgb24')
                video_dict = probe_video(video_path)
                break
        if video is None:
            print(f"No video found for {save_dir.name}")
            
        
        # Load the predictions
        csvs = list(save_dir.rglob('*.csv'))
        if not csvs:    
            print(f"No tracking CSV files found in {save_dir}")
            return
        
        print(f"Found {len(csvs)} tracking CSV files")
        zarr_files = list(save_dir.rglob('*.zarr'))
        assert len(zarr_files) == 1, "Expected exactly one predictions zarr file"
        zarr_file = zarr_files[0]
        print(f"Found predictions zarr file: {zarr_file}")
        # Load zarr
        store = zarr.storage.LocalStore(zarr_file, read_only=False)
        root = zarr.open_group(store=store, mode='a')
        print("Existing keys in zarr archive:", list(root.array_keys()))
        
        # Color handling
        # Recreate the colors here for the masks 
        # This is a bit of a hack, but it works if the n_labels and n_colors_submap 
        # match the original parameters used to create the colormap. 
        # So, if we don't change these parameters, the colors will be the same, because 
        # we are looking up object IDs from the original model classes
        all_labels_submaps = create_label_colors(n_labels=10,
                                                n_colors_submap=50
                                                )
        indices_max_diff_labels = sample_maximally_different(list(range(10)))
        indices_max_diff_subcolors = sample_maximally_different(list(range(50)))
        
        track_df_dict = {}
        color_dict = {}
        label_trackid_dict = {}
        label_counter_dict = {} # This is to find the right color in all_labels_submaps

        # Load the tracking data
        for csv in save_dir.glob('*.csv'):
            
            try:
                track_df = pd.read_csv(csv, header=5)
                assert len(track_df.label.unique()) == 1, "Multiple labels found in tracking data"  
                assert len(track_df.track_id.unique()) == 1, "Multiple track_ids found in tracking data"  
                label = track_df.iloc[0].label
                track_id = track_df.iloc[0].track_id
                label_trackid_dict[label] = track_id
            except (AttributeError, pd.errors.ParserError):
                print("Error reading .csv file. Trying old format ... ")
                track_df = pd.read_csv(csv)
                assert len(track_df.label.unique()) == 1, "Multiple labels found in tracking data"  
                assert len(track_df.track_id.unique()) == 1, "Multiple track_ids found in tracking data"  
                label = track_df.iloc[0].label
                track_id = track_df.iloc[0].track_id
                label_trackid_dict[label] = track_id
            
            # Check zarr
            assert f'{track_id}_masks' in list(root.array_keys()), "Mask not found in zarr archive" 
            mask_zarr = root[f'{track_id}_masks']
            max_frame_number = mask_zarr.shape[0] # This should be the same across all masks
            mask_shape = mask_zarr.shape[1:]
            classes = mask_zarr.attrs.get('classes', None) # These are the original classes from the model
            original_class_id = _find_class_by_name(classes, label)
            # Check if an index was already given for the current label
            # This is to extract the right color from the colormap
            if label not in label_counter_dict:
                label_counter_dict[label] = {'track_counter' : 0}
            else:
                label_counter_dict[label]['track_counter'] += 1
            # Find color 
            obj_color =  all_labels_submaps[indices_max_diff_labels[int(original_class_id)]]\
                [indices_max_diff_subcolors[label_counter_dict[label]['track_counter']]]
           
            mask_colors = DirectLabelColormap(color_dict={None: [0.,0.,0.,0.], 1: obj_color}, 
                                                    use_selection=True, 
                                                    selection=1,
                                                    )
                
            color_dict[int(track_id)] = mask_colors
            track_df_dict[int(track_id)] = track_df
            assert f'{track_id}_masks' in list(root.array_keys()), "Mask not found in zarr archive" 
            mask_zarr = root[f'{track_id}_masks']
            
        viewer = napari.Viewer()    
        if video is not None:
            add_layer = getattr(viewer, "add_image")
            layer_dict = {'name': video_path.stem,
                        'metadata': video_dict,   
                        }
            add_layer(video, **layer_dict)
        else:
            add_layer = getattr(viewer, "add_image")
            layer_dict = {'name': 'dummy mask'}
            add_layer(np.zeros((mask_shape)), **layer_dict)
        
        min_frame_number = 0
        # Loop over each track and add it to the viewer
        for track_id_to_plot in track_df_dict.keys():
            
            label = track_df_dict[track_id_to_plot].iloc[0].label   

            track_df_napari = track_df_dict[track_id_to_plot][['track_id','frame','pos_y','pos_x']].copy()
            features_df_napari = track_df_dict[track_id_to_plot][['frame','confidence','area','eccentricity','orientation','solidity']].copy()

            check_continuous = np.all(np.diff(track_df_napari['frame']) == 1)   
            if not check_continuous:
                print("Frames are not continuous ... interpolating")
                complete_tracking   = pd.DataFrame({'frame': range(int(min_frame_number), int(max_frame_number))})
                complete_features   = pd.DataFrame({'frame': range(int(min_frame_number), int(max_frame_number))})
                # Merge with existing data to identify missing frames
                merged_df_tracking = complete_tracking.merge(track_df_napari, on='frame', how='left')
                merged_df_tracking['track_id'] = track_id_to_plot
                merged_df_features = complete_features.merge(features_df_napari, on='frame', how='left')
                merged_df_features.fillna(0, inplace=True)
                # Make sure frame and track_id are the right types for interpolation
                merged_df_tracking['frame'] = merged_df_tracking['frame'].astype(int)
                merged_df_tracking['track_id'] = merged_df_tracking['track_id'].astype(int)

                # Interpolate the position columns
                pos_cols = ['pos_x', 'pos_y']
                for col in pos_cols:
                    merged_df_tracking[col] = merged_df_tracking[col].interpolate(method='linear')

                # # For any remaining NaN values at the start or end, use forward/backward fill
                #merged_df_tracking = merged_df_tracking.ffill()
                #merged_df_tracking = merged_df_tracking.bfill()
                # After interpolation, get the valid frame indices from the tracking DataFrame
                valid_frames = merged_df_tracking.dropna()['frame'].values
                # Use these valid frames to filter both DataFrames
                merged_df_tracking = merged_df_tracking[merged_df_tracking['frame'].isin(valid_frames)].reset_index(drop=True)
                merged_df_features = merged_df_features[merged_df_features['frame'].isin(valid_frames)].reset_index(drop=True)
                # Double-check that the frames match
                assert np.array_equal(merged_df_tracking['frame'].values, 
                                      merged_df_features['frame'].values),\
                                          "Frame mismatch between tracking and features"
                   
                track_df_napari = merged_df_tracking.copy()
                features_df_napari = merged_df_features.copy()
                # Before adding to Napari, change the column order
                track_df_napari = track_df_napari[['track_id', 'frame', 'pos_y', 'pos_x' ]]
                
            # Smooth the positions
            pos_cols = ['pos_x', 'pos_y']
            if sigma_tracking_pos > 0:
                for col in pos_cols:
                    track_df_napari[col] = gaussian_filter1d(track_df_napari[col], sigma=sigma_tracking_pos)
                
            for col in features_df_napari.columns:
                features_df_napari[col] = features_df_napari[col].astype(float)
            features_dict = features_df_napari.to_dict(orient='list')
            viewer.add_tracks(track_df_napari.values, 
                            features=features_dict,
                            blending='translucent', 
                            name=f'{label} - id {track_id_to_plot}', 
                            colormap='hsv',
                        )
            viewer.layers[f'{label} - id {track_id_to_plot}'].tail_width = 3
            viewer.layers[f'{label} - id {track_id_to_plot}'].tail_length = len(track_df_napari)
            viewer.layers[f'{label} - id {track_id_to_plot}'].color_by = 'frame'
            # Add masks
            mask_zarr = root[f'{track_id_to_plot}_masks']
            _ = viewer.add_labels(
                mask_zarr,
                name=f'{label} - MASKS - id {track_id_to_plot}',  
                opacity=0.5,
                blending='translucent',  
                colormap=color_dict[track_id_to_plot], 
                visible=[True if video is None else False][0],
            )
            
            viewer.dims.set_point(0,0)
            yield label, track_id, track_df_napari, features_df_napari, mask_zarr