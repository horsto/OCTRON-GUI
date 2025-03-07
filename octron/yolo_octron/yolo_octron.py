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


from octron.yolo_octron.helpers.yolo_checks import check_yolo_models
from octron.yolo_octron.helpers.polygons import (find_objects_in_mask, 
                                                 watershed_mask,
                                                 get_polygons,
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
    """
    
    def __init__(self, 
                 models_yaml_path,
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
        
        self.models_yaml_path = Path(models_yaml_path) 
        if not self.models_yaml_path.exists():
            raise FileNotFoundError(f"Model YAML file not found: {self.models_yaml_path}")

        # Check YOLO models, download if needed
        self.models_dict = check_yolo_models(YOLO_BASE_URL=None,
                                             models_yaml_path=self.models_yaml_path,
                                             force_download=False
                                             )
        
        # Set up internal variables
        self._project_path = None  # Use private variable for property
        self.training_path = None
        self.data_path = None
        self.model = None
        self.label_dict = None
        self.config_path = None
        
        # If a project path was provided, set it through the property setter
        if project_path is not None:
            self.project_path = project_path  # Uses the property setter
            
            # Setup training directories after project_path is validated
            self._setup_training_directories(self.clean_training_dir)
    
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


    def _setup_training_directories(self, clean_training_dir=False):
        """
        Setup folders for training
        
        Parameters
        ----------
        clean_training_dir : bool
            Whether to clean the training directory if it's not empty
        """
        if self._project_path is None:
            raise ValueError("Project path must be set before setting up training directories")
            
        # Setup folders for training
        self.training_path = self._project_path / 'model'  # Path to all model output
        self.data_path = self.training_path / 'training_data'  # Path to training data
        
        # Folder checks
        try:
            self.training_path.mkdir(exist_ok=False)
        except FileExistsError:
            # Check if training data folder is empty
            if len(list(self.training_path.glob('*'))) > 0:
                if not clean_training_dir:
                    raise FileExistsError(
                        f'"{self.training_path.as_posix()}" is not empty. Please remove subfolders first.')
                else:
                    shutil.rmtree(self.training_path)
                    self.training_path.mkdir()
                    print(f'Created fresh training directory "{self.training_path.as_posix()}"')
                    
                    
                    
                    
    ##### TRAINING DATA PREPARATION ###########################################################################    
    def prepare_labels(self, 
                       prune_empty_labels=True, 
                       min_num_frames=10, 
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
        The watershedding is performed on the masks,
        and the polygons are extracted from the resulting labels.
        
        I am doing some kind of "optimal" watershedding here,
        by determining the median object diameter from a subset of masks.
        This is then used to determine the footprint diameter for the watershedding.
        
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
        if self.label_dict is None:
            raise ValueError("No labels found. Please run prepare_labels() first.")

        for no_entry, labels in enumerate(self.label_dict.values(), start=1):  
            min_area = None

            for entry in labels:
                if entry == 'video':
                    continue
                label = labels[entry]['label']
                frames = labels[entry]['frames']
                mask_arrays = labels[entry]['masks'] # zarr array
                
                # On a subset of masks, determine object properties
                random_frames = pick_random_frames(frames, n=10)
                obj_diameters = []
                for f in random_frames:
                    for mask_array in mask_arrays:
                        sample_mask = mask_array[f]
                        if sample_mask.sum() == 0:
                            continue
                        else:
                            if min_area is None:
                                # Determine area threshold once
                                # threshold at 0.1 percent of the image area
                                min_area = 0.001*sample_mask.shape[0]*sample_mask.shape[1]
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

                ##################################################################################
                polys = {} # Collected polygons over frame indices
                for f_no, f in tqdm(enumerate(frames, start=1), 
                            desc=f'Polygons for label {label}', 
                            total=len(frames),
                            leave=True
                            ):    
                    mask_polys = [] # List of polygons for the current frame
                    for mask_array in mask_arrays:
                        mask_current_array = mask_array[f]
                        # Watershed
                        _, water_masks = watershed_mask(mask_current_array,
                                                        footprint_diameter=median_obj_diameter,
                                                        min_size_ratio=0.1,    
                                                        plot=False
                                                    )
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
                if entry == 'video':
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
                if entry == 'video':
                    continue
                assert 'frames' in labels[entry], "No frame indices (frames) found in labels"
                assert 'polygons' in labels[entry], "No polygons found in labels, run prepare_polygons() first"
                assert 'frames_split' in labels[entry], "No data split found in labels, run prepare_split() first"  

        # Create the training root directory
        # If it already exists, delete it and create a new one
        try:
            self.data_path.mkdir(exist_ok=False)
        except FileExistsError:
            shutil.rmtree(self.data_path)    
            self.data_path.mkdir()

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
            
            for entry in tqdm(labels,
                            total=len(labels),
                            position=0,
                            leave=True,
                            desc=f'Exporting {len(labels)} labels'
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
                if entry == 'video':
                    continue   
                if entry in label_id_label_dict:
                    assert label_id_label_dict[entry] == labels[entry]['label'],\
                        f"Label mismatch for {entry}: {label_id_label_dict[entry]} vs {labels[entry]['label']}"
                else:
                    label_id_label_dict[entry] = labels[entry]['label']

        ######## Write the YAML config
        self.config_path = self.data_path / "yolo_config.yaml"

        # Create the config dictionary
        config = {
            "path": str(dataset_path),
            "train": train_path,
            "test": test_path,
            "val": val_path,
        }
        config["names"] = label_id_label_dict
        header = "# OCTRON training dataset\n# Exported on {}\n\n".format(datetime.now())
        
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
        self.yolo_settings.update({
            'sync': False,
            'hub': False,
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
            
        self.model = YOLO(model_name_path)
        print(f"Model loaded from '{model_name_path.as_posix()}'")
        
        return self.model
    

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
                    mask_ratio=4,
                    epochs=epochs,
                    imgsz=imagesz,
                    resume=False,
                    plots=True,
                    batch=16,
                    cache=False,
                    save=True,
                    save_period=save_period, 
                    exist_ok=True,
                    # Augmentation
                    hsv_v=.25,
                    degrees=180,
                    scale=.5,
                    shear=2,
                    flipud=.1,
                    fliplr=.1,
                    mosaic=1.0,
                    copy_paste=.5,
                    copy_paste_mode='mixup', 
                    erasing=.25,
                    crop_fraction=1.0,
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
            
    
    def launch_tensorboard(self, port=6006):
        """
        Check if TensorBoard is installed, launch it with the training directory,
        and open a web browser to view the TensorBoard interface.
        
        Parameters
        ----------
        port : int
            Port number for TensorBoard to run on (default: 6006)
        
        Returns
        -------
        bool
            True if TensorBoard was successfully launched, False otherwise
        """
        
        
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
    
    
    def predict(self):
        # # Run inference on 'bus.jpg' with arguments
        # model.predict('/Users/horst/Downloads/octron_project/test data/8_behaviour_filtered2024-11-04T14_20_34_20240930_Th19.mp4', 
        #               save=True, 
        #               classes=[0],
        #               imgsz=1000, 
        #               device='cpu',
        #               visualize=False,
        #               conf=0.9
        #               )
        pass
