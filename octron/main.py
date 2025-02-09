'''
OCTRON
Main GUI file

'''
import os, sys
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import warnings

from pathlib import Path
cur_path  = Path(os.path.abspath(__file__)).parent.parent
base_path = Path(os.path.dirname(__file__)) # Important for example for .svg files
sys.path.append(cur_path.as_posix()) 

from qtpy.QtWidgets import (
    QWidget,
    QDialog,
    QApplication,
    QStyleFactory,
)
import napari
from napari.utils.notifications import (
    show_info,
    show_warning,
    show_error,
)
from napari.qt.threading import thread_worker
from napari.utils import DirectLabelColormap

# GUI 
from octron.gui_elements import octron_gui_elements

# SAM2 specific 
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
from octron.sam2_octron.helpers.build_sam2_octron import build_sam2_octron  
from octron.sam2_octron.helpers.sam2_checks import check_model_availability
from octron.sam2_octron.helpers.sam2_zarr import create_image_zarr

# Layer creation tools
from octron.sam2_octron.helpers.sam2_layer import (
    add_sam2_mask_layer,
    add_sam2_shapes_layer,
    add_sam2_points_layer,
)                

# Layer callbacks class
from octron.sam2_octron.helpers.sam2_layer_callback import sam2_octron_callbacks

# Object organizer
from octron.sam2_octron.object_organizer import Obj, ObjectOrganizer
  
# Custom dialog boxes
from octron.gui_dialog_elements import (
    add_new_label_dialog,
    remove_label_dialog,
)

# If there's already a QApplication instance (as may be the case when running as a napari plugin),
# then set its style explicitly:
app = QApplication.instance()
if app is not None:
    # This is a hack to get the style to look similar on darwin and windows systems
    # for the ToolBox widget
    app.setStyle(QStyleFactory.create("Fusion")) 



warnings.filterwarnings("ignore", category=FutureWarning)

class octron_widget(QWidget):
    '''
    
    '''
    def __init__(self, viewer: 'napari.viewer.Viewer', parent=None):
        super().__init__(parent)
        base_path_parent = base_path # TODO: Get rid of this path madness
        self.base_path = Path(os.path.abspath(__file__)).parent
        self._viewer = viewer
        self.remove_all_layers() # Aggressively delete all pre-existing layers in the viewer ...🪦 muahaha
        
        # Initialize some variables
        self.video_layer = None
        self.video_zarr = None
        self.prefetcher_worker = None
        self.predictor, self.device = None, None
        self.object_organizer = ObjectOrganizer() # Initialize top level object organizer
        
        # ... and some parameters
        self.chunk_size = 15 # Global parameter valid for both creation of zarr array and batch prediction 
        
        # Model yaml for SAM2
        models_yaml_path = self.base_path / 'sam2_octron/models.yaml'
        self.models_dict = check_model_availability(SAM2p1_BASE_URL='',
                                                    models_yaml_path=models_yaml_path,
                                                    force_download=False,
                                                    )
        
        # Initialize all UI components
        octron_gui = octron_gui_elements(self)
        octron_gui.setupUi(base_path=base_path_parent)
        
        # (De)activate certain functionality while WIP 
        # TODO
        last_index = self.layer_type_combobox.count() - 1
        self.layer_type_combobox.model().item(last_index).setEnabled(False)
                
        # Populate SAM2 dropdown list with available models
        for model_id, model in self.models_dict.items():
            print(f"Adding model {model_id}")
            self.sam2model_list.addItem(model['name'])
            
        # Connect (global) GUI callbacks 
        self.gui_callback_functions()
        # Connect layer specific callbacks
        self.octron_sam2_callbacks = sam2_octron_callbacks(self)

    ###################################################################################################
    
    def gui_callback_functions(self):
        '''
        Connect all callback functions to buttons and lists in the main GUI
        '''
        # Global layer insertion callback
        self._viewer.layers.events.inserted.connect(self.consolidate_layers)
        
        # Buttons 
        self.load_model_btn.clicked.connect(self.load_model)
        self.create_annotation_layer_btn.clicked.connect(self.create_annotation_layers)
        self.predict_next_batch_btn.clicked.connect(self.predict_next_batch)
        
        # Lists
        self.label_list_combobox.currentIndexChanged.connect(self.on_label_change)
    
    
    ###### SAM2 SPECIFIC CALLBACKS ####################################################################
    def load_model(self):
        '''
        Load the selected SAM2 model and enable the batch prediction button, 
        setting the progress bar to the chunk size and the button text to predict next chunk size
        
        '''
        index = self.sam2model_list.currentIndex()
        if index == 0:
            show_warning("Please select a valid model.")
            return
    
        model_name = self.sam2model_list.currentText()
        # Reverse lookup model_id
        for model_id, model in self.models_dict.items():
            if model['name'] == model_name:
                break
        
        print(f"Loading model {model_id}")
        model = self.models_dict[model_id]
        config_path = Path(model['config_path'])
        checkpoint_path = self.base_path / Path(f"sam2_octron/{model['checkpoint_path']}")
        self.predictor, self.device = build_sam2_octron(config_file=config_path.as_posix(),
                                                        ckpt_path=checkpoint_path.as_posix(),
                                                        )
        self.predictor.is_initialized = False
        show_info(f"Model {model_name} loaded on {self.device}")
        # Deactivate the dropdown menu upon successful model loading
        self.sam2model_list.setEnabled(False)
        self.load_model_btn.setEnabled(False)
        self.load_model_btn.setStyleSheet('QPushButton {background-color: #999; color: #495c10;}')
        self.load_model_btn.setText(f'{model_name} ✓')

        # Enable the predict next batch button
        # Take care of chunk size for batch prediction
        self.batch_predict_progressbar.setMaximum(self.chunk_size)
        self.batch_predict_progressbar.setValue(0)
        self.predict_next_batch_btn.setText(f'▷ Predict next {self.chunk_size} frames')
        self.predict_next_batch_btn.setEnabled(True)
        # TODO: Implement the predict next batch function
        #self.predict_next_batch_btn.clicked.connect(FUNCTION)
        # Check if you can create a zarr store for video
        self.create_video_zarr_prefetcher()

    ###### NAPARI SPECIFIC CALLBACKS ##################################################################

    def remove_all_layers(self):
        '''
        Remove all layers from the napari viewer.
        '''
        if len(self._viewer.layers):
            self._viewer.layers.select_all()
            self._viewer.layers.remove_selected()
            print("💀 Auto-deleted all old layers")
            
    
    def consolidate_layers(self, event):
        """
        Callback triggered when a layers are changed in the viewer.
        Currently triggered only on INSERTION events (layers are added).
        
        
        Takes care of defining video layers.
        Searches for video layers with a basename of "Image" and "VIDEO" in
        the name. If more than one is found, it removes the most recently added one.
        If one video layer is found and it contains metadata for num_frames, height,
        and width, it attaches a dummy mask to its metadata.
        """
        
        layer_name = event.value
        print(f"New layer added >>> {layer_name}")

        # Search through viewer layers for video layers with the expected criteria
        video_layers = []
        # Loop through all layers and check if they are video layers (TODO: Or others...)
        for l in self._viewer.layers:
            try:
                if l._basename() == 'Image' and 'VIDEO' in l.name:
                    video_layers.append(l)
            except Exception as e:
                show_error(f"💀 Error when checking layer: {e}")
        if len(video_layers) > 1:
            # TODO:For some reason this runs into an error when trying to remove the layer
            show_error("💀 More than one video layer found; Remove the extra one.")
            # Remove the most recently added video layer (or adjust as desired)
            #self._viewer.layers.remove(video_layers[-1])
        elif len(video_layers) == 1:
            video_layer = video_layers[0]
            #print(f"Found video layer >>> {video_layer.name}")
            # Check if required metadata exists before creating the dummy mask
            required_keys = ['num_frames', 'height', 'width']
            if all(k in video_layer.metadata for k in required_keys):
                mask_layer_dummy = np.zeros((
                    video_layer.metadata['num_frames'],
                    video_layer.metadata['height'],
                    video_layer.metadata['width']
                ), dtype=np.uint8)
                video_layer.metadata['dummy'] = mask_layer_dummy
                self.video_layer = video_layer
                self._viewer.dims.set_point(0,0)
                # Check if you can create a zarr store for video
                self.create_video_zarr_prefetcher()
            else:
                show_error("Video layer metadata incomplete; dummy mask not created.")
        else:
            pass

    def predict_next_batch(self):
        '''
        Thread worker for predicting the next batch of images
        
        '''
        print('Thread predicting ... WIP')
        # assert self.predictor, "No model loaded."
        # assert self.predictor.is_initialized, "Model not initialized."
        # self.batch_predictor = self.octron_sam2_callbacks.thread_predict()
        # self.batch_predictor.start()
            

    def create_video_zarr_prefetcher(self):
        '''
        This function deals with storage (temporary and long term).
        Long term: Zarr store
        Short term: Prefetcher worker
        
        ...
        Create a zarr store for the video layer.
        This will only work if a video layer is found and a model is loaded, 
        since both video and model information are required to create the zarr store.
        # TODO: Tie this into project management
        '''
        
        if self.video_zarr:
            # Zarr store already exists
            return
        if not self.video_layer:
            # only when a video layer is found
            return
        if not self.predictor:
            # only when a model is loaded
            return
        
        sample_dir = cur_path / 'sample_data'
        sample_dir.mkdir(exist_ok=True)
        sample_data_zarr = sample_dir / 'sample_data.zip'

        self.video_zarr = create_image_zarr(sample_data_zarr,
                                            num_frames=self.video_layer.metadata['num_frames'],
                                            image_height=self.predictor.image_size,
                                            chunk_size=self.chunk_size,
                                            )
        print(f'💾 Created video zarr archive under "{sample_data_zarr}"')
        # Set up thread worker to deal with prefetching batches of images
        self.prefetcher_worker = self.thread_prefetch_images()   
        self.prefetcher_worker.setAutoDelete(False)
        
        
    @thread_worker
    def thread_prefetch_images(self):
        '''
        Thread worker for prefetching images for fast processing in the viewer
        '''
        assert self.predictor, "No model loaded."
        assert self.predictor.is_initialized, "Model not initialized."
        current_indices = self._viewer.dims.current_step
        print(f'⚡️ Prefetching {self.chunk_size} images, start: {current_indices[0]}')
        _ = self.predictor.images[slice(current_indices[0],current_indices[0]+self.chunk_size)]

   
    def on_label_change(self):
        '''
        Callback function for the label list combobox.
        Handles the selection of labels, adding new labels, and removing labels. 
        
        '''
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
   
   
   
    def create_annotation_layers(self):
        '''
        Callback function for the create annotation layer button.
        Creates a new annotation layer based on the selected label and layer type.
        
        
        '''
        # First check if a model has been loaded
        if not self.predictor:
            show_warning("Please load a SAM2 model first.")
            return
        if not self.video_layer:
            show_warning("No video layer found.")
            return
        
        # Prewarm SAM2 predictor (model)
        # This needs the zarr store to be initialized first
        # -> that happens when either the model is loaded or a video layer is found
        # -> on_changed_layer() and load_model() take care of this
        if not self.predictor.is_initialized:
            self.predictor.init_state(video_data=self.video_layer.data, 
                                      zarr_store=self.video_zarr,
                                     )
            self.predictor.is_initialized = True
            
        # Get the selected label and layer type
        label = self.label_list_combobox.currentText().strip()
        # Get text from label suffix box
        label_suffix = self.label_suffix_lineedit.text()
        label_suffix = label_suffix.strip().lower() # Make sure things are somehow unified    
        
        layer_type = self.layer_type_combobox.currentText().strip()
        label_idx_ = self.label_list_combobox.currentIndex()
        layer_idx_ = self.layer_type_combobox.currentIndex()     
        if layer_idx_ == 0 or label_idx_ <= 2:
            show_warning("Please select a layer type and a label.")
            return
        
        # Start creating the new layer
        new_layer_name = f"{label} {label_suffix}".strip()
                
        # Optimistically find out what the next object ID should be and add to the object organizer
        obj_id = self.object_organizer.max_id() + 1
        self.object_organizer.add_entry(obj_id, Obj(label=label, suffix=label_suffix))
        
        new_organizer_entry = self.object_organizer.entries[obj_id] # Re-fetch to add more later
        obj_color = new_organizer_entry.color
        
        ######### Create a new mask layer  ######################################################### 
        
        mask_layer_name = f"{new_layer_name} masks"
        mask_colors = DirectLabelColormap(color_dict={None: [0.,0.,0.,0.], 1: obj_color}, 
                                          use_selection=True, 
                                          selection=1,
                                          )
        mask_layer = add_sam2_mask_layer(self._viewer,
                                         self.video_layer,
                                         mask_layer_name,
                                         mask_colors,
                                         )
        # For each layer that we create, write the object ID and the name to the metadata
        mask_layer.metadata['_name']   = mask_layer_name # Octron convention. Save a copy of the name
        mask_layer.metadata['_obj_id'] = obj_id # Save the object ID
        new_organizer_entry.mask_layer = mask_layer

        ######### Create a new annotation layer ####################################################
        
        if layer_type == 'Shapes':
            annotation_layer_name = f"{new_layer_name} shapes"
            # Create a shape layer
            annotation_layer = add_sam2_shapes_layer(self._viewer,
                                                     name=annotation_layer_name,
                                                     color=obj_color,
                                                     )
            # For each layer that we create, write the object ID and the name to the metadata
            annotation_layer.metadata['_name']   = mask_layer_name # Octron convention. Save a copy of the name
            annotation_layer.metadata['_obj_id'] = obj_id # Save the object ID
            new_organizer_entry.annotation_layer = annotation_layer
            # Connect callback
            annotation_layer.events.data.connect(self.octron_sam2_callbacks.on_shapes_changed)
            show_info(f"Created new mask + annotation layer '{new_layer_name}'")
            
            
        elif layer_type == 'Points':
            # Create a point layer
            annotation_layer_name = f"{new_layer_name} points"
            # Create a shape layer
            annotation_layer = add_sam2_points_layer(self._viewer,
                                                     name=annotation_layer_name,
                                                     )
            # For each layer that we create, write the object ID and the name to the metadata
            annotation_layer.metadata['_name']   = mask_layer_name # Octron convention. Save a copy of the name
            annotation_layer.metadata['_obj_id'] = obj_id # Save the object ID
            new_organizer_entry.annotation_layer = annotation_layer
            # Connect callback
            annotation_layer.mouse_drag_callbacks.append(self.octron_sam2_callbacks.on_mouse_press)
            annotation_layer.events.data.connect(self.octron_sam2_callbacks.on_points_changed)
            show_info(f"Created new mask + annotation layer '{new_layer_name}'")
            
            
        
        
        
        ######## Start prefetching images ##########################################################
        self.prefetcher_worker.start()
        self.label_list_combobox.setCurrentIndex(0)
        self.layer_type_combobox.setCurrentIndex(0)

            
            
               
    # Example key binding with Napari built-in viewer functions 
    # @viewer.bind_key('m')
    # def print_message(viewer):
    #    show_info('Test - pressed key m')
    

def octron_gui():
    '''
    This is the main entry point for the GUI call
    defined in the pyproject.toml file as 
    #      [project.gui-scripts]
    #      octron-gui = "octron.main:octron_gui"

    '''
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