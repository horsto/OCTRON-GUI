"""
OCTRON
Main GUI file

"""
import os, sys
from typing import List, Optional
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import shutil
from datetime import datetime
import warnings
# Suppress specific warnings
warnings.simplefilter("ignore")

from pathlib import Path
cur_path  = Path(os.path.abspath(__file__)).parent.parent
base_path = Path(os.path.dirname(__file__)) # Important for example for .svg files
sys.path.append(cur_path.as_posix()) 

from importlib.metadata import version
__version__ = version("octron")
octron_version = __version__

# Napari plugin QT components
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget,
    QDialog,
    QApplication,
    QStyleFactory,
    QFileDialog,
    QMessageBox,
    QHeaderView,
)
import napari
from napari.utils.notifications import (
    show_info,
    show_warning,
    show_error,
)
from napari.qt import create_worker
from napari.utils import DirectLabelColormap

# Napari PyAV reader 
from napari_pyav._reader import FastVideoReader

# GUI 
from octron.gui_elements import octron_gui_elements
from octron.gui_tables import ExistingDataTable

# SAM2 specific 
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
from octron.sam2_octron.helpers.video_loader import probe_video, get_vfile_hash
from octron.sam2_octron.helpers.build_sam2_octron import build_sam2_octron  
from octron.sam2_octron.helpers.sam2_checks import check_sam2_models
from octron.sam2_octron.helpers.sam2_zarr import (
    create_image_zarr,
    load_image_zarr,
)
# YOLO specific 
from octron.yolo_octron.gui.yolo_handler import YoloHandler
from octron.yolo_octron.helpers.training import collect_labels, load_object_organizer
from octron.yolo_octron.yolo_octron import YOLO_octron

# Tracker specific 
from octron.tracking.helpers.tracker_checks import check_boxmot_trackers
from octron.tracking.helpers.tracker_vis import create_color_icon

# Annotation layer creation tools
from octron.sam2_octron.helpers.sam2_layer import (
    add_sam2_mask_layer,
    add_sam2_shapes_layer,
    add_sam2_points_layer,
    add_annotation_projection,
)                

# Layer callbacks classes
from octron.sam2_octron.helpers.sam2_layer_callback import sam2_octron_callbacks

# OCTRON Object organizer
from octron.sam2_octron.object_organizer import Obj, ObjectOrganizer
  
# Custom dialog boxes
from octron.gui_dialog_elements import (
    add_new_label_dialog,
    remove_label_dialog,
)

# If there's already a QApplication instance (as may be the case when running as a napari plugin),
# then set its style explicitly:
# Enable high DPI support
if hasattr(Qt, 'AA_EnableHighDpiScaling'):
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
app = QApplication.instance()
if app is not None:
    # This is a hack to get the style to look similar on darwin and windows systems
    # for the ToolBox widget
    app.setStyle(QStyleFactory.create("Fusion")) 


class octron_widget(QWidget):
    """
    Main OCTRON widget class.
    It contains SAM2 methods for now. 
    All YOLO methods are in the YoloHandler class, to be found in yolo_octron/gui/yolo_handler.py   
    """

    def __init__(self, viewer: 'napari.viewer.Viewer', parent=None):
        super().__init__(parent)
        base_path_parent = base_path # TODO: Get rid of this path madness
        self.base_path = Path(os.path.abspath(__file__)).parent
        self._viewer = viewer
        self.app = QApplication.instance()
        self.remove_all_layers() # Aggressively delete all pre-existing layers in the viewer ...🪦 muahaha
        
        # Initialize some variables
        self.project_path = None # Main project path that the user selects
        self.project_path_video = None # Video path that the user selects
        self.video_layer = None
        self.current_video_hash = None # Hashed video file
        self.video_zarr = None
        self.all_zarrs = [] # Collect zarrs in list so they can be cleaned up upon closing
        self.prefetcher_worker = None
        self.predictor, self.device = None, None
        self.object_organizer = ObjectOrganizer() # Initialize top level object organizer
        self.remove_current_layer = False # Removal of layer yes/no
        self.layer_to_remove_idx = None # Index of layer to remove
        self.layer_to_remove = None # The actual layer to remove
        # ... and some parameters
        self.chunk_size = 15 # Global parameter valid for both creation of zarr array and batch prediction 
                             # For zarr arrays I set the minimum to ~50 frames for now
        self.skip_frames = 1 # Skip frames for prefetching images
    
        # Model yaml for SAM2
        sam2models_yaml_path = self.base_path / 'sam2_octron/sam2_models.yaml'
        self.sam2models_dict = check_sam2_models(SAM2p1_BASE_URL='',
                                                 models_yaml_path=sam2models_yaml_path,
                                                 force_download=False,
                                                 )
        # Model yaml for YOLO
        yolo_models_yaml_path = self.base_path / 'yolo_octron/yolo_models.yaml'
        self.yolo_octron = YOLO_octron(models_yaml_path=yolo_models_yaml_path) # Feeding in yaml to initiate models dict
        self.yolomodels_dict = self.yolo_octron.models_dict 
        
        # Model yaml for Trackers
        trackers_yaml_path = self.base_path / 'tracking/boxmot_trackers.yaml'
        self.trackers_dict = check_boxmot_trackers(trackers_yaml_path)
        
        # Initialize all UI components
        octron_gui_elements(self, base_path=base_path_parent)
        
        # Initialize sub GUI handlers for YOLO
        self.yolo_handler = YoloHandler(self, self.yolo_octron)
        self.yolo_handler.connect_signals()
        
        # (De)activate certain functionality while WIP 
        last_index = self.layer_type_combobox.count() - 1
        self.layer_type_combobox.model().item(last_index).setEnabled(False)
        
        # Populate SAM2 dropdown list with available models
        for model_id, model in self.sam2models_dict.items():
            print(f"Adding SAM2 model {model_id}")
            self.sam2model_list.addItem(model['name'])
            
        # Populate YOLO dropdown list with available models
        for model_id, model in self.yolomodels_dict.items():
            print(f"Adding YOLO model {model_id}")
            self.yolomodel_list.addItem(model['name'])
   
        # Populate tracker list with available boxmot trackers
        for tracker in self.trackers_dict:
            if self.trackers_dict[tracker]['available']:
                print(f'Adding tracker {self.trackers_dict[tracker]["name"]}')
                color = self.trackers_dict[tracker]['color']
                square_icon = create_color_icon(color) # Creates color icon to show computational demands
                self.yolomodel_tracker_list.addItem(square_icon, self.trackers_dict[tracker]['name']+ " ")

        # Connect (global) GUI callbacks 
        self.gui_callback_functions()
        # Connect layer specific callbacks
        self.octron_sam2_callbacks = sam2_octron_callbacks(self)
        print(f'OCTRON GUI v{octron_version} initialized')

    ###################################################################################################
    
    def gui_callback_functions(self):
        """
        Connect all callback functions to buttons and lists in the main GUI
        """
        # Global layer insertion callback
        self._viewer.layers.events.inserted.connect(self.consolidate_layers)
        
        # Global layer removal callback
        self._viewer.layers.events.removing.connect(self.on_layer_removing)
        self._viewer.layers.events.removed.connect(self.on_layer_removed)
        
        # Main video drop area
        self.video_file_drop_widget.callback = self.on_mp4_file_dropped_area

        # Buttons 
        # ... project 
        self.create_project_btn.clicked.connect(self.open_project_folder_dialog)
        # ... SAM2 and annotations 
        self.load_sam2model_btn.clicked.connect(self.load_sam2model)
        self.create_annotation_layer_btn.clicked.connect(self.create_annotation_layers)
        self.predict_next_batch_btn.clicked.connect(self.init_prediction_threaded)
        self.predict_next_oneframe_btn.clicked.connect(self.init_prediction_threaded)    
        self.create_projection_layer_btn.clicked.connect(self.create_annotation_projections)
        self.annotation_jump_previous_btn.clicked.connect(self.jump_to_previous_annotated_frame)
        self.annotation_jump_next_btn.clicked.connect(self.jump_to_next_annotated_frame)
        self.hard_reset_layer_btn.clicked.connect(self.reset_predictor)
        self.hard_reset_layer_btn.setEnabled(False)
        # ... YOLO
        self.generate_training_data_btn.setText('')
        self.start_stop_training_btn.setText('')
        self.predict_start_btn.setText('')
        # Lists
        self.label_list_combobox.currentIndexChanged.connect(self.on_label_change)
        # Upon start, disable some of the toolbox tabs and functionality for video drop 
        self.project_video_drop_groupbox.setEnabled(False)
        self.toolBox.widget(1).setEnabled(False) # Annotation
        self.toolBox.widget(2).setEnabled(False) # Training
        self.toolBox.widget(3).setEnabled(False) # Prediction
        
        # Disable layer annotation until SAM2 model is loaded
        self.annotate_layer_create_groupbox.setEnabled(False)
        
        # And ... 
        self.train_generate_groupbox.setEnabled(False)
        self.train_train_groupbox.setEnabled(False)
        # ... 
        self.predict_video_drop_groupbox.setEnabled(False)
        self.predict_video_predict_groupbox.setEnabled(False)
        
        # Connect to the Napari viewer close event
        self.app.lastWindowClosed.connect(self.closeEvent)
    

    ###### SAM2 SPECIFIC CALLBACKS ####################################################################
    
    def load_sam2model(self, 
                       model_name='',
                       ):
        """
        Load the selected SAM2 model and enable the batch prediction button, 
        setting the progress bar to the chunk size and the button text to predict next chunk size
        
        Parameters
        ----------
        model_name : str
            The name of the SAM2 model to load. If None, the currently selected model in the dropdown list is used.
            If no model is selected, the function returns without doing anything.
        
        """
        if not model_name:
            # Assuming this is retrievable from current GUI ...             
            index = self.sam2model_list.currentIndex()
            if index == 0:
                return
            model_name = self.sam2model_list.currentText()
        # Reverse lookup model_id
        model_found = False
        for model_id, model in self.sam2models_dict.items():
            if model['name'] == model_name:
                model_found = True
                break
        assert model_found, f"Model '{model_name}' not found in SAM2 models dictionary."
        
        print(f"Loading SAM2 model {model_id}")
        model = self.sam2models_dict[model_id]
        config_path = Path(model['config_path'])
        checkpoint_path = self.base_path / Path(f"sam2_octron/{model['checkpoint_path']}")
        self.predictor, self.device = build_sam2_octron(config_file_path=config_path.as_posix(),
                                                        ckpt_path=checkpoint_path.as_posix(),
                                                        )
        self.predictor.is_initialized = False
        show_info(f"SAM2 model {model_name} loaded on {self.device}")
        # Deactivate the dropdown menu upon successful model loading
        self.sam2model_list.setEnabled(False)
        self.load_sam2model_btn.setEnabled(False)
        self.load_sam2model_btn.setText(f'{model_name} ✓')

        # Enable the predict next batch button
        # Take care of chunk size for batch prediction
        self.batch_predict_progressbar.setMaximum(self.chunk_size)
        self.batch_predict_progressbar.setValue(0)
        
        self.predict_next_batch_btn.setText(f'▷ {self.chunk_size} frames')
        self.predict_next_oneframe_btn.setText('▷')
        self.predict_next_oneframe_btn.setEnabled(True)
        self.predict_next_batch_btn.setEnabled(True)
        
        # Check if you can create a zarr store for video
        # Creating a zarr store for the video is only possible if a video has been loaded
        # AND a model has been loaded
        self.init_zarr_prefetcher_threaded()
        # Enable the annotation layer creation tab
        self.annotate_layer_create_groupbox.setEnabled(True)
        
        
    def reset_predictor(self):
        """
        Reset the predictor and all layers.
        """
        self.predictor.reset_state()
        annotation_layers = self.object_organizer.get_annotation_layers()
        for layer in annotation_layers:
            layer.data = []
        show_info("SAM2 predictor was reset.")
        
    
    def _batch_predict_yielded(self, value):
        """
        prediction_worker()
        Called upon yielding from the batch prediction thread worker.
        Updates the progress bar and the mask layer with the predicted mask.
        """
        progress, frame_idx, obj_id, mask, last_run = value
        organizer_entry = self.object_organizer.get_entry(obj_id)
        # Extract current mask layer
        prediction_layer = organizer_entry.prediction_layer
        prediction_layer.data[frame_idx,:,:] = mask
        prediction_layer.refresh()  
        if self._viewer.dims.current_step[0] != frame_idx and not last_run:
            self._viewer.dims.set_point(0, frame_idx)
        self.batch_predict_progressbar.setValue(progress)
          
    def _on_prediction_finished(self):
        """
        prediction_worker()
        Callback for when worker within init_prediction_threaded() 
        has finished executing. 
        """
        # Enable the predcition button again
        self.predict_next_batch_btn.setEnabled(True)
        self.predict_next_oneframe_btn.setEnabled(True)
        self.skip_frames_spinbox.setEnabled(True)
        self.batch_predict_progressbar.setValue(0)
        
        # Save the object organizer and also refresh the table view
        # ! TODO: Make this more efficient. This slows down everything a lot since 
        # what we are doing here is creating a video hash from scratch twice (?) and 
        # load the video data, plus we find out which indices have annotation data in the 
        # video. So, that is a lot of processing for just refreshing the table view for example 
        self.save_object_organizer()
        self.refresh_label_table_list(delete_old=False) # This is the table in the project tab
        self.batch_predict_progressbar.setMaximum(self.chunk_size)    

    def init_prediction_threaded(self):
        """
        Thread worker for predicting the next batch of images
        """
        
        # Before doing anything, make sure, some input has been provided
        valid = False
        if (self.predictor.inference_state['point_inputs_per_obj'] or 
            self.predictor.inference_state['mask_inputs_per_obj']):
            valid = True
        if not valid:
            show_warning("Please annotate at least one object first.")
            return

        # Identify the sender (button) that called this function
        sender = self.sender()
        
        # Disable the prediction button
        self.predict_next_batch_btn.setEnabled(False)
        self.predict_next_oneframe_btn.setEnabled(False)
        self.skip_frames_spinbox.setEnabled(False)
        if sender == self.predict_next_batch_btn:
            self.prediction_worker = create_worker(self.octron_sam2_callbacks.batch_predict)
            self.prediction_worker.setAutoDelete(True)
            self.prediction_worker.yielded.connect(self._batch_predict_yielded)
            self.prediction_worker.finished.connect(self._on_prediction_finished)
            self.prediction_worker.start()
        elif sender == self.predict_next_oneframe_btn:
            self.batch_predict_progressbar.setMaximum(1)
            self.prediction_worker_one = create_worker(self.octron_sam2_callbacks.next_predict)
            self.prediction_worker_one.setAutoDelete(True)
            self.prediction_worker_one.yielded.connect(self._batch_predict_yielded)
            self.prediction_worker_one.finished.connect(self._on_prediction_finished)
            self.prediction_worker_one.start()
        

    ###### NAPARI SPECIFIC CALLBACKS ##################################################################
    
    def closeEvent(self):
        """
        Callback for the Napari viewer close event
        """
        
        for zarr_store in self.all_zarrs:
            if zarr_store is not None:
                store = zarr_store.store
                if hasattr(store, 'close'):
                    store.close()
                    print(f"Zarr {zarr_store} store closed.")
        # Clean up the prefetcher worker
        if self.prefetcher_worker is not None:
            self.prefetcher_worker.quit()
        # Lastly, save object organizer to json 
        if self.project_path:
            self.save_object_organizer()
            
    def save_object_organizer(self):
        """
        Save the object organizer to the project directory
        """
        if self.project_path_video is None or not self.project_path_video.exists():
            print("No project video path set or found. Not exporting object organizer.")
            return
        organizer_path = self.project_path_video  / "object_organizer.json"
        self.object_organizer.save_to_disk(organizer_path)

    def refresh_label_table_list(self, delete_old=False):
        """
        Refresh the label list combobox with the current labels in the object organizer
        
        Parameter
        ----------
        delete_old : bool
            If True, delete the old entries from the combobox
        
        """
        # Check this folder for existing project data
        label_dict = collect_labels(self.project_path, 
                                    subfolder=self.current_video_hash, 
                                    prune_empty_labels=False,  
                                    min_num_frames=0
                                    )  
        if not label_dict:
            return
        
        # Initialize the table model if not already done
        if not hasattr(self, 'label_table_model'):
            self.label_table_model = ExistingDataTable()
            self.existing_data_table.setModel(self.label_table_model)
            
            # Configure the table appearance
            # Prevent resizing of table columns by disabling interactive resizing
            header = self.existing_data_table.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.Fixed)
            header.setSectionsMovable(False)
            header.setStretchLastSection(True)
            # Connect double-click event on table rows 
            self.existing_data_table.doubleClicked.connect(self.on_label_table_double_clicked)
        
        # Update the table with new data
        self.label_table_model.update_data(label_dict, delete_old=delete_old)
        
        # Enable the groupbox for existing data and the training generation box
        self.project_existing_data_groupbox.setEnabled(True)
        self.train_generate_groupbox.setEnabled(True)
        self.generate_training_data_btn.setStyleSheet('QPushButton { color: #8ed634;}')
        self.generate_training_data_btn.setText(f'▷ Generate')
        
        # Enable training tab if data is available
        if label_dict and any(v for k, v in label_dict.items() if k != 'video' and k != 'video_file_path'):
            print("Data available, enabling training tab.")
            self.toolBox.widget(2).setEnabled(True)  # Training
            self.train_generate_groupbox.setEnabled(True)
            self.train_train_groupbox.setEnabled(True)
            # Enable some buttons too 
            self.train_data_watershed_checkBox.setEnabled(True)
            self.train_data_overwrite_checkBox.setEnabled(True)
            self.train_prune_checkBox.setEnabled(True)

    def on_label_table_double_clicked(self, index):
        """
        Handle double-click events on the existing data table.
        This is the ExistingDataTable() instance.
        
        Parameters
        ----------
        index : QModelIndex
            The index of the clicked cell
        """
        # Get the full folder path from the model
        folder_path = self.label_table_model.get_folder_path(index)
        if not folder_path:
            return

        # Load the video from the folder path
        video_file_path = Path(self.label_table_model.label_dict[folder_path].get('video_file_path'))
        if not video_file_path.exists():
            show_warning("No video file found in folder.")
            return
        
        # Remove the current video layer
        # This triggers a bunch of things -> check function for details
        if self.video_layer is not None:
            self._viewer.layers.remove(self.video_layer)
            
        # Reset variables for a clean start
        self.object_organizer = ObjectOrganizer()
        # SAM2 
        self.sam2model_list.setEnabled(True)
        self.load_sam2model_btn.setEnabled(True)
        self.load_sam2model_btn.setText(f'Load model')
        self.predict_next_batch_btn.setText('')
        self.predict_next_oneframe_btn.setText('')
        self.predict_next_oneframe_btn.setEnabled(False)
        self.predict_next_batch_btn.setEnabled(False)
            
        # Use the file drop method to load the video
        self.on_mp4_file_dropped_area([video_file_path])
        
        # Re-instantiate all layers
        # Find .json 
        folder_path = Path(folder_path)
        json_files = list(folder_path.glob('*.json'))
        if not json_files:
            show_warning("No .json files found in folder.")
            return
        if len(json_files) > 1:
            show_warning("More than one .json file found in folder.")
            return
        json_file = json_files[0]
        # Load the object organizer from the .json file
        object_organizer_data = load_object_organizer(json_file)
        if not object_organizer_data:
            show_warning("Could not load object organizer data.")
            return
        
        # Clear the label list combobox and re-initialize it
        self.label_list_combobox.clear()
        self.label_list_combobox.addItem("Label ...")
        self.label_list_combobox.addItem(u"\u2295 Create")
        self.label_list_combobox.addItem(u"\u2296 Remove")
        
        # Track unique labels to avoid duplicates
        unique_labels = set()
        # Loop over all objects in the object organizer data
        # and re-create the layers
        for obj_id, obj_data in object_organizer_data['entries'].items():
            print(f"Loading object with ID {obj_id}")
            obj_id = int(obj_id)
            label = obj_data.get('label', '')
            suffix = obj_data.get('suffix', '')
            obj_color = obj_data.get('color', [0.7, 0.7, 0.7, 1])
            
            # Add label to combobox if not already added
            if label and label not in unique_labels:
                unique_labels.add(label)
                self.label_list_combobox.addItem(label)
            
            # Determine layer type from metadata
            layer_type = None
            if 'annotation_layer_metadata' in obj_data:
                annotation_type = obj_data['annotation_layer_metadata'].get('type')
                if annotation_type == 'Shapes':
                    layer_type = 'Shapes'
                elif annotation_type == 'Points':
                    layer_type = 'Points'
            if not layer_type:  
                layer_type = 'Points' # Default to Points if not specified
             # Create layers with the reconstructed information
            print(f"Recreating layer: {label} {suffix} (ID: {obj_id}, Type: {layer_type})")
            self.create_annotation_layers(
                recreate=True,
                label=label,
                layer_type=layer_type,
                label_suffix=suffix,
                obj_id=obj_id,
                obj_color=obj_color
            )
            
            
        # Reset the combobox to the first item
        self.label_list_combobox.setCurrentIndex(0)    
        return
    
    def set_project_folder(self, folder):
        """
        Set (new) OCTRON project folder. 
        Makes sure existing video layers (and thereby all other layers)
        are removed from the viewer.
        Enables the video drop area and the project folder path label.
        Refreshes the label table list and the YOLO model list.
                
        Parameters
        ----------
        folder : str or Path
            The path to the project folder. 
            This should be an existing folder that contains the OCTRON project data.
    
        """
        
        folder = Path(folder)
        assert folder.exists(), f"Project folder {folder} does not exist."
        
        # Remove the current video layer
        # This triggers a bunch of things -> check function "on_layer_removed()"
        if self.video_layer is not None:
            self._viewer.layers.remove(self.video_layer)
            
        # Reset variables for a clean start
        self.object_organizer = ObjectOrganizer()
        self.project_folder_path_label.setEnabled(False)
        self.project_folder_path_label.setText(f'→{folder.as_posix()}')
        
        self.project_path = folder
        self.project_video_drop_groupbox.setEnabled(True)
        self.refresh_label_table_list(delete_old=True)
        self.yolo_handler.refresh_trained_model_list()
        return

    def open_project_folder_dialog(self):
        """
        Open a file dialog for the user to choose a base folder for the current OCTRON project.
        This yields self.project_path, which is used to store all project-related data.
        
        If the project path is an existing OCTRON project, it will be inspected and 
        the existing annotation data will be quickly displayed in a table view.
        
        """
        # Open a directory selection dialog
        folder = QFileDialog.getExistingDirectory(self, "Select Base Folder", str(Path.home()))
        if folder:
            print(f"Project base folder selected: {folder}")
            self.set_project_folder(folder)            
        else:
            print("No folder selected.")
        return 

    def remove_all_layers(self, spare=[]):
        """
        Remove all layers from the napari viewer.
        
        Parameters
        ----------
        spare : list
            List of layers that should not be removed
        """
        if not isinstance(spare, list):
            spare = [spare]
        print(f'🗑️  Deleting all layers except "{spare}"')
        # First remove mask layers (to avoid dependencies with annotation layers)
        mask_layers = []
        other_layers = []
        
        # Categorize layers
        for layer in list(self._viewer.layers):
            if layer in spare:
                continue
            elif 'masks' in layer.name.lower():
                mask_layers.append(layer)
            else:
                other_layers.append(layer)
        # Remove mask layers first
        for layer in mask_layers:
            if layer in self._viewer.layers:
                self._viewer.layers.remove(layer)
        # Then remove other layers
        for layer in other_layers:
            if layer in self._viewer.layers:
                self._viewer.layers.remove(layer)
                
        total_deleted = len(mask_layers) + len(other_layers)
        if total_deleted:
            print(f"💀 Auto-deleted {total_deleted} layers")


    def on_layer_removed(self, event):
        """
        Callback triggered from within the layer removal event.
        (self.on_layer_removing() is called first)
        This gives the user a chance to cancel the removal of the layer.
        It also organizes the removal process according to which layer type is being removed. 
        
        """        
        if not self.remove_current_layer:
            # TODO: This is a bit of a hack, seems ugly. Is there a better way?
            new_old_layer = self._viewer.add_layer(self.layer_to_remove)
            self._viewer.layers.selection.active = new_old_layer
            self._viewer.layers.selection.active.mode = 'pan_zoom'
        else:
            print(f"❌ Removed layer {self.layer_to_remove.name}")
            # What else do you need to remove? 
            # Three cases:
            # 1. The layer is a mask layer
            # 2. The layer is an annotation layer
            # 3. The layer is a video layer 
            
            # 1. Mask layer
            if self.layer_to_remove._basename() == 'Labels' \
                and 'mask' in self.layer_to_remove.metadata['_name']:
                # Remove the zarr zip file containing the layer data
                # zarr_file_path = self.project_path / self.layer_to_remove.metadata['_zarr']
                # if Path(zarr_file_path).exists():
                #     shutil.rmtree(zarr_file_path)
                #     print(f'Removed Zarr file {zarr_file_path}')
                # Get the object entry from the object organizer
                obj_id = self.layer_to_remove.metadata['_obj_id']
                organizer_entry = self.object_organizer.get_entry(obj_id)
                annotation_layer = organizer_entry.annotation_layer
                # Remove the object entry from the object organizer
                self.object_organizer.remove_entry(obj_id)
                # Remove obj_id from current SAM2 predictor
                try:
                    self.predictor.remove_object(obj_id, strict=True)
                    print(f"Removed object with ID{obj_id} from organizer and predictor")
                except (RuntimeError, AttributeError) as e:
                    print(f"Error when removing object from SAM2 predictor: {e}")
                    print("This is likely due to the SAM2 model not being loaded and can be ignored.")
                
                # Finally, trigger removal of the annotation layer
                if annotation_layer is not None:
                    self._viewer.layers.remove(annotation_layer)
            # 2. Annotation layer
            elif self.layer_to_remove._basename() in ['Shapes', 'Points']:
                # Get the object entry from the object organizer
                obj_id = self.layer_to_remove.metadata['_obj_id']
                organizer_entry = self.object_organizer.get_entry(obj_id)
                if organizer_entry is not None:
                    organizer_entry.annotation_layer = None
            # 3. Video layer
            # This has more drastic consequences ...
            elif self.layer_to_remove._basename() == 'Image' and 'VIDEO' in self.layer_to_remove.name:
                # What to do: 
                # Remove all layers and reset SAM predictor
                self.remove_all_layers(spare=self.layer_to_remove)
                # Also re-instantiate variables
                self.project_path_video = None
                self.video_layer = None 
                self.current_video_hash = None
                self.video_zarr = None
                if hasattr(self, 'prefetcher_worker') and self.prefetcher_worker is not None:
                    self.prefetcher_worker.quit()
                self.prefetcher_worker = None
                self.all_zarrs = []
                # SAM2 
                self.predictor = None   
                self.sam2model_list.setEnabled(True)
                self.load_sam2model_btn.setEnabled(True)
                self.load_sam2model_btn.setText(f'Load model')
                self.predict_next_batch_btn.setText('')
                self.predict_next_oneframe_btn.setText('')
                self.predict_next_oneframe_btn.setEnabled(False)
                self.predict_next_batch_btn.setEnabled(False)
                # Object organizer
                self.object_organizer = ObjectOrganizer()
                # Disable the layer annotation box until SAM2 is loaded 
                self.annotate_layer_create_groupbox.setEnabled(False)
                # Reset naming of annotation tab 
                self.toolBox.setItemText(1, "Generate annotation data")
            # Reset the flag 
            self.remove_current_layer = False
    
        return
    
    
    def on_layer_removing(self, event):
        """ 
        Callback triggered when a layer is about to be removed.
        The call to on_layer_removed() is triggered after this, and gives 
        the user a chance to cancel the removal of the layer.
        """
        self.layer_to_remove = event.source[event.index]
        self.layer_to_remove_idx = event.index  
        # Not sure if possible to delete more than one at once.
        # If so, then take care of it ... event.sources is a list.
        if self.layer_to_remove._basename() in ['Shapes', 'Points', 'Image', 'Labels']:
            # Silent removal of annotation layers
            self.remove_current_layer = True
        # elif self.layer_to_remove._basename() == 'Image': #and 'VIDEO' not in self.layer_to_remove.name:
        #     # Silent removal of image layers (visualizations)
        #     self.remove_current_layer = True
        
        # LEAVE THIS IN HERE, it can be useful in the future. 
        # else:
        #     # Ask for confirmation for other layers, i.e. mask layers
        #     reply = QMessageBox.question(
        #         None, 
        #         "Confirmation", 
        #         f"Are you sure you want to delete layer\n'{self.layer_to_remove}'",
        #         QMessageBox.Yes | QMessageBox.No,
        #         QMessageBox.No
        #     )
        #     if reply == QMessageBox.No:
        #         self.remove_current_layer = False
        #     else:
        #         self.remove_current_layer = True
        return
          
    
    def consolidate_layers(self, event):
        """
        Callback triggered when a layers are changed in the viewer.
        Currently triggered only on INSERTION events (layers are added).
        
        Takes care of defining video layers.
        Searches for video layers with a basename of "Image" and "VIDEO" in
        the name. 


        Sets
        ----
        self.video_layer : napari.layers.Image
            The current video layer
        self.current_video_hash : str   
            The hash of the current video layer
        self.video_zarr : zarr.storage
            The zarr storage for the video layer (via the prefetcher)
        self.all_zarrs : list
            List of all zarr stores for the video layer (via the prefetcher)
        self.prefetcher_worker : QThread
            The worker thread for the prefetcher (via the prefetcher)
        self.project_path_video : Path
            The path to the video project directory
        """
        
        layer_name = event.value
        
        # Search through viewer layers for video layers with the expected criteria
        # Starting here as an example with video layers, but 
        # this could be anything in the future ... let's see if we need it 
        video_layers = []
        # Loop through all layers and check if they are video layers
        for l in self._viewer.layers:
            try:
                if l._basename() == 'Image' and 'VIDEO' in l.name:
                    video_layers.append(l)
            except Exception as e:
                show_error(f"💀 Error when checking layer: {e}")

        if len(video_layers) > 1:
            # This should never happen
            show_warning("🙀 More than one video layer found. Skipping.")
            return
        if not video_layers:
            return
        
        # If there is only one video layer, set it as the current video layer
        video_layer = video_layers[0]    
        if self.video_layer == video_layer:
            return
        else:
            self.video_layer = video_layer 
            video_metadata = video_layer.metadata
            self.current_video_hash = video_metadata['hash'][-8:] # Save it globally for quick access       
            self.project_path_video = self.project_path / self.current_video_hash
            self._viewer.dims.set_point(0,0)
            
            # Check if you can create a zarr store for video
            # Creating a zarr store for the video is only possible if a video has been loaded
            # AND a model has been loaded
            self.init_zarr_prefetcher_threaded()
            
            print(f"VIDEO LAYER >>> {layer_name}")
            self.toolBox.widget(1).setEnabled(True) 
            self.toolBox.setItemText(1, f"Generate annotation data for: {self.current_video_hash}")

        return
        
    def on_mp4_file_dropped_area(self, video_paths):
        """
        Adds video layer on freshly dropped mp4 file.
        Callback function for the file drop area in the project management tab.
        The area itself (a widget) is already filtering for mp4 files.
        """
        # Check if project path is set
        if self.video_layer is not None:
            show_warning("A video layer already exists. Please remove it first.")
            return
        if not self.project_path:
            show_error("Please select a project directory first.")
            return
    
        if len(video_paths) > 1:
            show_warning("Please drop only one file at a time.")
            return
        
        video_path = Path(video_paths[0]) # Take only the first file if there are multiple
        # Load video file and meta info
        if not video_path.exists():
            show_error("File does not exist.")
            return
        # Check if the video is within the project directory or a subdirectory
        try:
            # Resolve to absolute paths to handle symlinks
            video_absolute = video_path.resolve()
            project_absolute = self.project_path.resolve()
            
            # Check if video path is within project path
            if not str(video_absolute).startswith(str(project_absolute)):
                # Video is not in project path, ask if user wants to copy it
                reply = QMessageBox.question(
                    None,
                    "Video Location",
                    f"The video file is outside the project directory.\n\n"
                    f"Video: {video_path}\n"
                    f"Project: {self.project_path}\n\n"
                    f"Would you like to copy the video to the project directory?",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                    QMessageBox.Yes
                )
                if reply == QMessageBox.Cancel:
                    return
                if reply == QMessageBox.Yes:
                    # Create videos directory if it doesn't exist
                    videos_dir = self.project_path / "videos"
                    videos_dir.mkdir(exist_ok=True)
                    # Copy video to project directory
                    new_video_path = videos_dir / video_path.name
                    # Show progress dialog for large files
                    if video_path.stat().st_size > 50_000_000:  # 50 MB
                        show_info(f"Copying large video file to project directory... Please wait.")
                    # Copy the file
                    shutil.copy2(video_path, new_video_path)
                    video_path = new_video_path
                    show_info(f"Video copied to {new_video_path}")
        
        except Exception as e:
            show_error(f"Error checking video path: {str(e)}")
            return
        
        video_dict = probe_video(video_path)
        # Create hash and save it in the metadata
        video_file_hash = get_vfile_hash(video_path)
        video_dict['hash'] = video_file_hash # Save it locally as metadata
        # Layer name 
        layer_name = f'VIDEO [name: {video_path.stem}]'
        video_dict['_name'] = layer_name # Octron convention. Save the name in metadata.
        layer_dict = {'name'    : layer_name,
                      'metadata': video_dict,
                     }
        add_layer = getattr(self._viewer, "add_image")
        add_layer(FastVideoReader(video_path, read_format='rgb24'), **layer_dict)

        
    def init_sam2_model(self):
        """
        Initialize the SAM2 model for the current session.
        This function requires a video to be loaded AND a model to be selected / loaded. 
        It is called only once from within the prefetcher worker.
        
        """
        if not self.predictor:
            show_warning("Please select a SAM2 model first.")
            return
        if not self.video_layer:
            print("No video layer found.")
            return
        if not self.video_zarr:
            show_warning("No video zarr store found.")
            return
        
        # Prewarm SAM2 predictor (model)
        # This needs the zarr store to be initialized first
        # -> that happens when either the model is loaded or a video layer is found
        # -> on_changed_layer() and load_sam2model() take care of this
        if not self.predictor.is_initialized:
            self.predictor.init_state(video_data=self.video_layer.data, 
                                      zarr_store=self.video_zarr,
                                      )
            self.hard_reset_layer_btn.setEnabled(True)
            self.predictor.is_initialized = True
            
        return
    
        
    def init_zarr_prefetcher_threaded(self):
        """
        This function deals with storage (temporary and long term).
        Long term: Zarr store
        Short term: Threaded prefetcher worker
        
        ...
        Create a zarr store for the video layer.
        This will only work if a video layer is found and a sam2 model is loaded, 
        since both video and model information are required to create the zarr store.

        """
        if not self.project_path:
            return
        if self.video_zarr:
            # Zarr store already exists
            return
        if not self.video_layer or not self.current_video_hash:
            # only when a video layer and corresponding hash are found
            return
        if not self.predictor:
            # only when a model is loaded
            return
        
        # Collect some info about video layer before loading or creating zarr archive
        metadata =  self.video_layer.metadata
        num_frames = metadata['num_frames']
        video_height = metadata['height']
        video_width = metadata['width']    
        predictor_image_size = self.predictor.image_size # SAM2 model image size
        largest_edge = max(video_height, video_width) 

        image_scaler = predictor_image_size / largest_edge
        # Resize both (!) edges to the same size 
        # This is a hack since I did not get SAM2 to work for non-square videos
        resized_height = int(np.floor(image_scaler * largest_edge)) 
        resized_width = int(np.floor(image_scaler * largest_edge))
        print(f'📐 Resized video dimensions: {resized_height}x{resized_width}')
        
        # Create zarr store for video layer
        
        video_zarr_path = self.project_path_video / 'video data.zarr'
        status = False
        
        if video_zarr_path.exists():
            # Zarr store already exists. Check and load. 
            # If the checks fail, the zarr store will be recreated.
            video_zarr, status = load_image_zarr(video_zarr_path,
                                                 num_frames=num_frames,
                                                 image_height=resized_height,
                                                 image_width=resized_width,
                                                 chunk_size=self.chunk_size,
                                                 num_ch=3,
                                                 video_hash_abrrev=self.current_video_hash,
                                                 )
                                            
        if not status or not video_zarr_path.exists():
            if video_zarr_path.exists():
                shutil.rmtree(video_zarr_path)
            self.video_zarr = create_image_zarr(video_zarr_path,
                                                num_frames=num_frames,
                                                image_height=resized_height,
                                                image_width=resized_width,
                                                chunk_size=self.chunk_size,
                                                num_ch=3,
                                                video_hash_abbrev=self.current_video_hash,
                                                )
            # Write video information to a text file
            video_info_path = self.project_path_video / "video_info.txt"
            with open(video_info_path, 'w') as f:
                f.write("# This info here is just for information purposes.\n")    
                f.write("# It is not actually used anywhere in OCTRON and can be deleted\n")   
                f.write("# or edited without consequences.\n") 
                f.write("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n") 
                f.write(f"Video path: {metadata['video_file_path']}\n")
                f.write(f"Video hash: {metadata['hash']}\n")
                f.write(f"Video abbreviated hash: {self.current_video_hash}\n") # Used throughout as identifier
                f.write(f"Number of frames: {num_frames}\n")
                f.write(f"Original resolution (hxw): {video_height}x{video_width}\n")
                f.write(f"Info file created on: {datetime.now()}\n")
            print(f'💾 New video zarr archive created "{video_zarr_path.as_posix()}"')
            print(f'📝 Video info saved to "{video_info_path.as_posix()}"')
        else:
            self.video_zarr = video_zarr
            print(f'📖 Video zarr archive loaded "{video_zarr_path.as_posix()}"')
        
        # Add to list of zarrs for cleanup upon closing
        self.all_zarrs.append(self.video_zarr)
        # Set up thread worker to deal with prefetching batches of images
        self.prefetcher_worker = create_worker(self.octron_sam2_callbacks.prefetch_images)
        self.prefetcher_worker.setAutoDelete(False)
        self.prefetcher_worker.start()
        
    
    def on_label_change(self):
        """
        Callback function for the label list combobox.
        Handles the selection of labels, adding new labels, and removing labels. 
        
        """
        index = self.label_list_combobox.currentIndex()
        current_text = self.label_list_combobox.currentText()
        all_list_entries = [self.label_list_combobox.itemText(i) for i in range(self.label_list_combobox.count())]
        if index == 0:
            return
        elif index == 1:
            # Add new label was selected by user
            dialog = add_new_label_dialog(self)
            dialog.exec_()
            if dialog.result() == QDialog.Accepted:
                new_label_name = dialog.label_name.text()
                new_label_name = new_label_name.strip().lower() # Make sure things are somehow unified
                if not new_label_name:
                    show_warning("Please enter a valid label name.")
                    self.label_list_combobox.setCurrentIndex(0)
                    return
                if new_label_name in all_list_entries:
                    # Select the existing label
                    existing_index = all_list_entries.index(new_label_name)
                    self.label_list_combobox.setCurrentIndex(existing_index)
                    show_warning(f'Label "{new_label_name}" already exists.')
                else:
                    self.label_list_combobox.addItem(new_label_name)
                    new_index = self.label_list_combobox.count()-1
                    self.label_list_combobox.setCurrentIndex(new_index)
                    show_info(f'Added new label "{new_label_name}"')
            else:
                self.label_list_combobox.setCurrentIndex(0)
                return
            
        elif index == 2:
            if len(all_list_entries) <= 3: 
                self.label_list_combobox.setCurrentIndex(0)
                return
             # User wants to remove a label
            dialog = remove_label_dialog(self, all_list_entries[3:])
            dialog.exec_()
            if dialog.result() == QDialog.Accepted:
                selected_label = dialog.list_widget.currentItem().text()
                self.label_list_combobox.removeItem(self.label_list_combobox.findText(selected_label))
                self.label_list_combobox.setCurrentIndex(0)
                show_info(f'Removed label "{selected_label}"')
            else:
                self.label_list_combobox.setCurrentIndex(0)
                return
        else:
            print(f'Selected label {current_text}')   
   
   
    def create_annotation_layers(self,
                                 recreate : bool = False,
                                 label: str = "",
                                 layer_type: str = "",
                                 label_suffix: str = "",
                                 obj_id: Optional[int] = None,
                                 obj_color: List[float] = [],
                                 ):
        """
        This is the main callback function for the create annotation layer button.
        Creates a new annotation layer based on the selected label 
        and layer type (recreate=False).
        It is also utilized to re-create layers from the object organizer when the user 
        double clicks on a table row (recreate=True). 
        
        Returns
        -------
        annotation_layer : napari.layers.Shapes or napari.layers.Points
            The created annotation layer, either a Shapes or Points layer.
            If the layer already exists, it will return the existing layer.
            If the layer could not be created, it will return None.
        prediction_layer : napari.layers.Labels
            The created prediction layer (mask layer).
            If the layer already exists, it will return the existing layer.
            If the layer could not be created, it will return None.
        organizer_entry : Obj
            The object entry in the object organizer that corresponds to the created layers.
        
        """
        # Check if a video layer is loaded
        if not self.video_layer:
            show_warning("No video layer found.")
            return
        if not self.project_path_video:
            show_warning("No project video path found.")
            return

        if not recreate:
            # Check if parameters were provided 
            if not label or not layer_type:
                # Sanity check for dropdown 
                # ... exclude the first entries (Choose, Add, Remove)
                label_idx_ = self.label_list_combobox.currentIndex()
                layer_idx_ = self.layer_type_combobox.currentIndex()     
                if layer_idx_ == 0 or label_idx_ <= 2:
                    show_warning("Please select a layer type and a label.")
                    return
                # Get the selected label and layer type from dropdowns
                label = self.label_list_combobox.currentText().strip()
                label_suffix = self.label_suffix_lineedit.text()
                label_suffix = label_suffix.strip().lower() # Make sure things are somehow unified    
                layer_type = self.layer_type_combobox.currentText().strip()
            else:
                # If parameters were provided, use them directly
                label = label.strip()
                label_suffix = label_suffix.strip().lower() if label_suffix else ""
                layer_type = layer_type.strip()
        
        # Check if the object organizer already has an entry for this label and suffix 
        organizer_entry = self.object_organizer.get_entry_by_label_suffix(label, label_suffix)
        if organizer_entry is not None:
            if organizer_entry.prediction_layer is None:
                # Should never happen!
                show_warning(f"Combination ({label}, {label_suffix}) exists. No mask layer found.")
                return
            elif organizer_entry.annotation_layer is not None:
                show_warning(f"Combination ({label}, {label_suffix}) already exists.")
                return
            else:
                create_prediction_layer = False
                obj_id = self.object_organizer.get_entry_id(organizer_entry)
        else:
            # No entry found, so create a new prediction and annotation layer from scratch
            create_prediction_layer = True
            if obj_id is None:
                obj_id = self.object_organizer.min_available_id()
            status = self.object_organizer.add_entry(obj_id, Obj(label=label, suffix=label_suffix))
            if not status: 
                show_error("Error when adding new entry to object organizer.")
                return
            organizer_entry = self.object_organizer.entries[obj_id] # Re-fetch to add more later
        
        ######### Create new layers #############################################################################
        
        if not obj_color:
            obj_color = organizer_entry.color # Get the color from the organizer entry

        layer_name = f"{label} {label_suffix}".strip() # This is used in all layer names
        
        ######### Create a new prediction (mask) layer  ######################################################### 
        
        if create_prediction_layer:
            prediction_layer_name = f"{layer_name} masks"
            mask_colors = DirectLabelColormap(color_dict={None: [0.,0.,0.,0.], 1: obj_color}, 
                                              use_selection=True, 
                                              selection=1,
                                              )
            prediction_layer, zarr_data, zarr_file_path = add_sam2_mask_layer(viewer=self._viewer,
                                                                 video_layer=self.video_layer,
                                                                 name=prediction_layer_name,
                                                                 project_path=self.project_path_video,
                                                                 color=mask_colors,
                                                                 video_hash_abrrev=self.current_video_hash,
                                                                 )
            # Add zarr store to list for cleanup upon closing
            self.all_zarrs.append(zarr_data)
            if prediction_layer is None:
                show_error("Error when creating mask layer.")
                return
            # For each layer that we create, write the object ID and the name to the metadata
            prediction_layer.metadata['_name']   = prediction_layer_name # Save a copy of the name
            prediction_layer.metadata['_obj_id'] = obj_id # This corresponds to organizer entry id
            prediction_layer.metadata['_zarr'] = zarr_file_path.relative_to(self.project_path)
            prediction_layer.metadata['_hash'] = self.current_video_hash
            try:
                # By default, try to extract a relative file path. 
                # This enables users to move the project folder around without breaking the link.
                prediction_layer.metadata['_video_file_path'] = Path(self.video_layer.metadata['video_file_path']).relative_to(self.project_path)
            except ValueError:
                # If the video file is not in a subdirectory of the project folder,
                # save the absolute path instead.
                prediction_layer.metadata['_video_file_path'] = Path(self.video_layer.metadata['video_file_path'])
            organizer_entry.prediction_layer = prediction_layer

        ######### Create a new annotation layer ###############################################################
        annotation_layer = None
        if layer_type == 'Shapes':
            annotation_layer_name = f"{layer_name} shapes"
            # Create a shape layer
            annotation_layer = add_sam2_shapes_layer(viewer=self._viewer,
                                                     name=annotation_layer_name,
                                                     color=obj_color,
                                                     )
            annotation_layer.metadata['_name']   = annotation_layer_name 
            annotation_layer.metadata['_obj_id'] = obj_id 
            annotation_layer.metadata['_hash'] = self.current_video_hash
            organizer_entry.annotation_layer = annotation_layer
            # Connect callback
            annotation_layer.events.data.connect(self.octron_sam2_callbacks.on_shapes_changed)
            print(f"Created new mask + annotation layer '{layer_name}'")
            
            
        elif layer_type == 'Points':
            # Create a point layer
            annotation_layer_name = f"{layer_name} points"
            # Create a shape layer
            annotation_layer = add_sam2_points_layer(viewer=self._viewer,
                                                     name=annotation_layer_name,
                                                     )
            annotation_layer.metadata['_name']   = annotation_layer_name 
            annotation_layer.metadata['_obj_id'] = obj_id 
            annotation_layer.metadata['_hash'] = self.current_video_hash
            organizer_entry.annotation_layer = annotation_layer
            # Connect callback
            annotation_layer.mouse_drag_callbacks.append(self.octron_sam2_callbacks.on_mouse_press)
            annotation_layer.events.data.connect(self.octron_sam2_callbacks.on_points_changed)
            print(f"Created new mask + annotation layer '{layer_name}'")
            
            
        else: 
            # Reserved space for anchor point layer here ... 
            pass
           
           
        # Reset the dropdowns
        self.label_list_combobox.setCurrentIndex(0)
        self.layer_type_combobox.setCurrentIndex(0)

        return annotation_layer, prediction_layer, organizer_entry
    

    def create_annotation_projections(self):
        """
        Create a projection layer for annotated label.
        """
        self.create_projection_layer_btn.setEnabled(False)  

        # Loop over all annotation labels and execute add_annotation_projection
        for label in self.object_organizer.get_current_labels():
            add_annotation_projection(self._viewer,
                                      self.object_organizer,
                                      label,
                                      )
        self.create_projection_layer_btn.setEnabled(True) 
        


    def jump_to_next_annotated_frame(self):
        """
        Jump to next annotated frame in viewer timeline
        """
        current_timeline_idx = self._viewer.dims.current_step[0]
        # Go through all prediction_layers and check if they have a mask for the current frame
        # If so, jump to the next frame with a mask
        prediction_layers = self.object_organizer.get_prediction_layers()
        if not prediction_layers:
            show_warning("No prediction layers found.")
            return
        
        indices = []
        for layer in prediction_layers:
            data = layer.data
            annotated_indices = np.where(data[:,0,0] >= 0)[0]
            # Get the next index after the current one
            next_idx = np.where(annotated_indices > current_timeline_idx)[0]
            if next_idx.size > 0:
                indices.append(annotated_indices[next_idx[0]])
        if not indices:
            show_warning("No further annotated frames found.")
            return
        else:
            next_idx = min(indices)
            self._viewer.dims.set_point(0, next_idx)
        return      
        
        
    def jump_to_previous_annotated_frame(self):
        """
        Jump to previous annotated frame in viewer timeline
        """
        current_timeline_idx = self._viewer.dims.current_step[0]
        # Go through all prediction_layers and check if they have a mask for the current frame
        # If so, jump to the next frame with a mask
        prediction_layers = self.object_organizer.get_prediction_layers()
        if not prediction_layers:
            return
        
        indices = []
        for layer in prediction_layers:
            data = layer.data
            annotated_indices = np.where(data[:,0,0] >= 0)[0]
            # Get the next index after the current one
            prev_idx = np.where(annotated_indices < current_timeline_idx)[0]
            if prev_idx.size > 0:
                indices.append(annotated_indices[prev_idx[-1]])
        if not indices:
            return
        else:
            prev_idx = max(indices)
            self._viewer.dims.set_point(0, prev_idx)
        return

###############################################################################################################
###############################################################################################################
###############################################################################################################

def octron_gui():
    """
    This is the main entry point for the GUI call
    defined in the pyproject.toml file as 
    #      [project.gui-scripts]
    #      octron-gui = "octron.main:octron_gui"

    """
    viewer = napari.Viewer()
    
    # If there's already a QApplication instance (as may be the case when running as a napari plugin),
    # then set its style explicitly:
    app = QApplication.instance()
    if app is not None:
        # This is a hack to get the style to look similar on darwin and windows systems
        # for the ToolBox widget
        app.setStyle(QStyleFactory.create("Fusion")) 
    
    viewer.window.add_dock_widget(octron_widget(viewer))
    napari.run()


if __name__ == "__main__":
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(octron_widget(viewer))
    napari.run()