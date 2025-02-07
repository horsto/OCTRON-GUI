'''
OCTRON
Main GUI file.

'''
import os, sys
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import warnings

from pathlib import Path
cur_path = Path(os.path.abspath(__file__)).parent.parent
print(f"Adding {cur_path.as_posix()} to sys.path")
sys.path.append(cur_path.as_posix())

from qtpy.QtCore import QSize, QRect, Qt, QCoreApplication
from qtpy.QtGui import QCursor, QPixmap, QIcon
from qtpy.QtWidgets import (
    QWidget,
    QDialog,
    QFrame,
    QApplication,
    QStyleFactory,
    QVBoxLayout,
    QLabel,
    QLayout,
    QToolBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QSizePolicy,
    QLineEdit,
    QSpinBox,
    QProgressBar,
    QPushButton,
    QAbstractSpinBox,
    QGridLayout
)
import napari
from napari.utils.notifications import (
    show_info,
    show_warning,
    show_error,
)


# SAM2 specific 
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
from octron.sam2_octron.helpers.build_sam2_octron import build_sam2_octron  
from octron.sam2_octron.helpers.sam2_checks import check_model_availability

# Layer creation tools
from octron.sam2_octron.helpers.sam2_layers import (
    add_sam2_mask_layer,
)                

# Color tools
from napari.utils import DirectLabelColormap
from octron.sam2_octron.helpers.sam2_colors import (
    create_label_colors,
)

# Custom dialog boxes
from octron.dialog import (
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
        self.cur_path = Path(os.path.abspath(__file__)).parent
        
        self._viewer = viewer
        self.remove_all_layers() # Aggressively delete all pre-existing layers in the viewer ...🪦 muahaha
        
        
        # Initialize some variables
        self.video_layer = None
        self.obj_id_label = {} # Object ID to label mapping
        self.obj_id_layer = {} # Object ID to layer mapping
        # ... and some parameters
        self.chunk_size = 15 # Global parameter valid for both creation of zarr array and batch prediction 
        
        # Model yaml for SAM2
        models_yaml_path = self.cur_path / 'sam2_octron/models.yaml'
        self.models_dict = check_model_availability(SAM2p1_BASE_URL='',
                                                    models_yaml_path=models_yaml_path,
                                                    force_download=False,
                                                    )
        self.predictor, self.device = None, None
        
        # Colors for the labels
        self.label_colors = create_label_colors(cmap='cmr.tropical')

        ##################################################################################################
        # Initialize all UI components
        self.setupUi()
        
        # (De)activate certain functionality while WIP 
        # TODO
        last_index = self.layer_type_combobox.count() - 1
        self.layer_type_combobox.model().item(last_index).setEnabled(False)
                
        # Populate SAM2 dropdown list with available models
        for model_id, model in self.models_dict.items():
            print(f"Adding model {model_id}")
            self.sam2model_list.addItem(model['name'])
            
        
        # Connect callbacks 
        self.callback_functions()
        
        
        
        

    def callback_functions(self):
        '''
        Connect all callback functions to buttons and lists 
        '''
        # Global layer insertion callback
        self._viewer.layers.events.inserted.connect(self.on_changed_layer)
        
        # Buttons 
        self.load_model_btn.clicked.connect(self.load_model)
        self.create_annotation_layer_btn.clicked.connect(self.create_annotation_layers)
        
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
        checkpoint_path = self.cur_path / Path(f"sam2_octron/{model['checkpoint_path']}")
        self.predictor, self.device = build_sam2_octron(config_file=config_path.as_posix(),
                                                        ckpt_path=checkpoint_path.as_posix(),
                                                        )
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
        
    
    ###### NAPARI SPECIFIC CALLBACKS ##################################################################

    def remove_all_layers(self):
        '''
        Remove all layers from the napari viewer.
        This is important because we cannot guarantee that 
        what has previously been loaded (maybe by accident)
        has gone through proper inspection.
        '''
        if len(self._viewer.layers):
            self._viewer.layers.select_all()
            self._viewer.layers.remove_selected()
            print("💀 Auto-deleted all old layers")
            
    
    def on_changed_layer(self, event):
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
        for l in self._viewer.layers:
            try:
                if l._basename() == 'Image' and 'VIDEO' in l.name:
                    video_layers.append(l)
            except Exception as e:
                show_error(f"💀 Error when checking layer: {e}")
        if len(video_layers) > 1:
            # TODO: 
            # For some reason this runs into an error when trying to remove the layer
            show_error("💀 More than one video layer found; Remove the extra one.")
            # Remove the most recently added video layer (or adjust as desired)
            #self._viewer.layers.remove(video_layers[-1])
        elif len(video_layers) == 1:
            video_layer = video_layers[0]
            print(f"Found video layer >>> {video_layer.name}")
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
            else:
                show_error("Video layer metadata incomplete; dummy mask not created.")
        else:
            show_warning("No video layer found.")
            
            
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
        
        # Get the selected label and layer type
        label = self.label_list_combobox.currentText().strip()
        label_idx = self.label_list_combobox.currentIndex()
        layer_type = self.layer_type_combobox.currentText().strip()
        layer_idx = self.layer_type_combobox.currentIndex()     
        if layer_idx == 0 or label_idx <= 2:
            show_warning("Please select a layer type and a label.")
            return
        
        # Get text from label suffix box
        label_suffix = self.label_suffix_lineedit.text()
        label_suffix = label_suffix.strip().lower() # Make sure things are somehow unified        
        
        label_name = f"{label} {label_suffix}".strip()
        # Check if the layer_name already exists in self.obj_id_label

        if label_name in self.obj_id_label.values():
            show_error(f"Layer with label '{label_name}' exists already.")
            return
            
        # At this point, the basic checks are complete.
        # Next, find out which obj_id should be used 
        if not self.obj_id_label:
            current_obj_id = 0
        else:
            current_obj_id = max(self.obj_id_label.keys()) + 1
            
        ######### Create a new mask layer  ######################################################### 
        mask_layer_name = f"{label_name} MASKS"
        base_color = DirectLabelColormap(color_dict=self.label_colors[0], 
                                         use_selection=True, 
                                         selection=current_obj_id,
                                         )
        mask_layer = add_sam2_mask_layer(viewer,
                                         self.video_layer,
                                         mask_layer_name,
                                         base_color,
                                         )
        mask_layer.metadata['_name'] = mask_layer_name # Octron convention. Save a copy of name
        self.obj_id_label[current_obj_id] = label_name
        self.obj_id_layer[current_obj_id] = mask_layer
        
        ######### Create a new annotation layer ####################################################
        
        
        
        if layer_type == 'Shape Layer':
            # Create a shape layer
            pass
        elif layer_type == 'Point Layer':
            # Create a point layer
            pass
            
        self.label_list_combobox.setCurrentIndex(0)
        self.layer_type_combobox.setCurrentIndex(0)
            
            
            
            
            
               
    # Example key binding with Napari built-in viewer functions 
    # @viewer.bind_key('m')
    # def print_message(viewer):
    #    show_info('Test - pressed key m')
    ###### GUI SETUP CODE FROM QT DESIGNER ############################################################
        
    def setupUi(self):
        if not self.objectName():
            self.setObjectName(u"self")
        self.setEnabled(True)
        self.resize(410, 700)
        self.setMinimumSize(QSize(410, 700))
        self.setMaximumSize(QSize(410, 800))
        self.setCursor(QCursor(Qt.ArrowCursor))
        self.setWindowOpacity(1.000000000000000)
        self.verticalLayoutWidget = QWidget(self)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QRect(0, 0, 412, 651))
        self.mainLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.mainLayout.setSpacing(10)
        self.mainLayout.setObjectName(u"mainLayout")
        self.mainLayout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.octron_logo = QLabel(self.verticalLayoutWidget)
        self.octron_logo.setObjectName(u"octron_logo")
        self.octron_logo.setEnabled(True)
        self.octron_logo.setMinimumSize(QSize(410, 100))
        self.octron_logo.setBaseSize(QSize(0, 0))
        self.octron_logo.setLineWidth(0)
        self.octron_logo.setPixmap(QPixmap(u"qt_gui/octron_logo.svg"))
        self.octron_logo.setAlignment(Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop)

        self.mainLayout.addWidget(self.octron_logo, 0, Qt.AlignmentFlag.AlignLeft)

        self.toolBox = QToolBox(self.verticalLayoutWidget)
        self.toolBox.setObjectName(u"toolBox")
        self.toolBox.setEnabled(True)
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.toolBox.sizePolicy().hasHeightForWidth())
        self.toolBox.setSizePolicy(sizePolicy)
        self.toolBox.setMinimumSize(QSize(0, 0))
        self.toolBox.setMaximumSize(QSize(410, 750))
        self.toolBox.setCursor(QCursor(Qt.ArrowCursor))
        self.toolBox.setFrameShape(QFrame.Shape.NoFrame)
        self.toolBox.setFrameShadow(QFrame.Shadow.Plain)
        self.toolBox.setLineWidth(0)
        self.toolBox.setMidLineWidth(0)
        self.project_tab = QWidget()
        self.project_tab.setObjectName(u"project_tab")
        self.project_tab.setGeometry(QRect(0, 0, 410, 387))
        sizePolicy1 = QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.project_tab.sizePolicy().hasHeightForWidth())
        self.project_tab.setSizePolicy(sizePolicy1)
        icon = QIcon()
        icon.addFile(u"qt_gui/icons/noun-project-7158867.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.toolBox.addItem(self.project_tab, icon, u"Project")
        self.annotate_tab = QWidget()
        self.annotate_tab.setObjectName(u"annotate_tab")
        self.annotate_tab.setGeometry(QRect(0, 0, 405, 387))
        sizePolicy1.setHeightForWidth(self.annotate_tab.sizePolicy().hasHeightForWidth())
        self.annotate_tab.setSizePolicy(sizePolicy1)
        self.annotate_tab.setMaximumSize(QSize(405, 700))
        self.verticalLayoutWidget_2 = QWidget(self.annotate_tab)
        self.verticalLayoutWidget_2.setObjectName(u"verticalLayoutWidget_2")
        self.verticalLayoutWidget_2.setGeometry(QRect(0, 0, 402, 391))
        self.annotate_vertical_layout = QVBoxLayout(self.verticalLayoutWidget_2)
#ifndef Q_OS_MAC
        self.annotate_vertical_layout.setSpacing(-1)
#endif
        self.annotate_vertical_layout.setObjectName(u"annotate_vertical_layout")
        self.annotate_vertical_layout.setContentsMargins(0, 0, 0, 10)
        self.horizontalGroupBox = QGroupBox(self.verticalLayoutWidget_2)
        self.horizontalGroupBox.setObjectName(u"horizontalGroupBox")
        sizePolicy2 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.horizontalGroupBox.sizePolicy().hasHeightForWidth())
        self.horizontalGroupBox.setSizePolicy(sizePolicy2)
        self.horizontalGroupBox.setMinimumSize(QSize(400, 60))
        self.horizontalGroupBox.setMaximumSize(QSize(400, 60))
        self.horizontalLayout_8 = QHBoxLayout(self.horizontalGroupBox)
        self.horizontalLayout_8.setSpacing(20)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalLayout_8.setContentsMargins(9, 9, 9, 9)
        self.sam2model_list = QComboBox(self.horizontalGroupBox)
        self.sam2model_list.addItem("")
        self.sam2model_list.setObjectName(u"sam2model_list")
        self.sam2model_list.setMinimumSize(QSize(167, 25))
        self.sam2model_list.setMaximumSize(QSize(167, 25))

        self.horizontalLayout_8.addWidget(self.sam2model_list, 0, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.load_model_btn = QPushButton(self.horizontalGroupBox)
        self.load_model_btn.setObjectName(u"load_model_btn")
        self.load_model_btn.setMinimumSize(QSize(0, 25))
        self.load_model_btn.setMaximumSize(QSize(250, 25))

        self.horizontalLayout_8.addWidget(self.load_model_btn, 0, Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)


        self.annotate_vertical_layout.addWidget(self.horizontalGroupBox, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop)

        self.annotate_layer_create_groupbox = QGroupBox(self.verticalLayoutWidget_2)
        self.annotate_layer_create_groupbox.setObjectName(u"annotate_layer_create_groupbox")
        sizePolicy2.setHeightForWidth(self.annotate_layer_create_groupbox.sizePolicy().hasHeightForWidth())
        self.annotate_layer_create_groupbox.setSizePolicy(sizePolicy2)
        self.annotate_layer_create_groupbox.setMinimumSize(QSize(400, 90))
        self.annotate_layer_create_groupbox.setMaximumSize(QSize(400, 90))
        self.gridLayout = QGridLayout(self.annotate_layer_create_groupbox)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.gridLayout.setHorizontalSpacing(20)
        self.gridLayout.setVerticalSpacing(9)
        self.gridLayout.setContentsMargins(9, 9, 9, 9)
        self.label_list_combobox = QComboBox(self.annotate_layer_create_groupbox)
        self.label_list_combobox.addItem("")
        self.label_list_combobox.addItem("")
        self.label_list_combobox.addItem("")
        self.label_list_combobox.setObjectName(u"label_list_combobox")
        self.label_list_combobox.setMinimumSize(QSize(110, 25))
        self.label_list_combobox.setMaximumSize(QSize(110, 25))
        self.label_list_combobox.setEditable(False)
        self.label_list_combobox.setMaxCount(15)
        self.label_list_combobox.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.label_list_combobox.setIconSize(QSize(14, 14))
        self.label_list_combobox.setFrame(False)

        self.gridLayout.addWidget(self.label_list_combobox, 0, 2, 1, 1, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.layer_type_combobox = QComboBox(self.annotate_layer_create_groupbox)
        self.layer_type_combobox.addItem("")
        self.layer_type_combobox.addItem("")
        self.layer_type_combobox.addItem("")
        self.layer_type_combobox.addItem("")
        self.layer_type_combobox.setObjectName(u"layer_type_combobox")
        self.layer_type_combobox.setMinimumSize(QSize(110, 25))
        self.layer_type_combobox.setMaximumSize(QSize(110, 25))
        self.layer_type_combobox.setMaxCount(15)
        self.layer_type_combobox.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.layer_type_combobox.setIconSize(QSize(14, 14))
        self.layer_type_combobox.setFrame(False)

        self.gridLayout.addWidget(self.layer_type_combobox, 0, 0, 1, 1, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.create_annotation_layer_btn = QPushButton(self.annotate_layer_create_groupbox)
        self.create_annotation_layer_btn.setObjectName(u"create_annotation_layer_btn")
        self.create_annotation_layer_btn.setMinimumSize(QSize(70, 25))
        self.create_annotation_layer_btn.setMaximumSize(QSize(70, 25))

        self.gridLayout.addWidget(self.create_annotation_layer_btn, 0, 4, 1, 1, Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)

        self.label_suffix_lineedit = QLineEdit(self.annotate_layer_create_groupbox)
        self.label_suffix_lineedit.setObjectName(u"label_suffix_lineedit")
        self.label_suffix_lineedit.setMinimumSize(QSize(60, 25))
        self.label_suffix_lineedit.setMaximumSize(QSize(60, 25))
        self.label_suffix_lineedit.setInputMask(u"")
        self.label_suffix_lineedit.setText(u"")
        self.label_suffix_lineedit.setMaxLength(100)

        self.gridLayout.addWidget(self.label_suffix_lineedit, 0, 3, 1, 1, Qt.AlignmentFlag.AlignVCenter)

        self.hard_reset_layer_btn = QPushButton(self.annotate_layer_create_groupbox)
        self.hard_reset_layer_btn.setObjectName(u"hard_reset_layer_btn")
        self.hard_reset_layer_btn.setMinimumSize(QSize(70, 25))
        self.hard_reset_layer_btn.setMaximumSize(QSize(70, 25))
        self.hard_reset_layer_btn.setAutoRepeatInterval(2000)

        self.gridLayout.addWidget(self.hard_reset_layer_btn, 1, 4, 1, 1, Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)

        self.create_projection_layer_btn = QPushButton(self.annotate_layer_create_groupbox)
        self.create_projection_layer_btn.setObjectName(u"create_projection_layer_btn")
        self.create_projection_layer_btn.setMinimumSize(QSize(110, 25))
        self.create_projection_layer_btn.setMaximumSize(QSize(110, 25))

        self.gridLayout.addWidget(self.create_projection_layer_btn, 1, 0, 1, 1, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)


        self.annotate_vertical_layout.addWidget(self.annotate_layer_create_groupbox, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop)

        self.annotate_param_groupbox = QGroupBox(self.verticalLayoutWidget_2)
        self.annotate_param_groupbox.setObjectName(u"annotate_param_groupbox")
        sizePolicy3 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        sizePolicy3.setHorizontalStretch(100)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.annotate_param_groupbox.sizePolicy().hasHeightForWidth())
        self.annotate_param_groupbox.setSizePolicy(sizePolicy3)
        self.annotate_param_groupbox.setMinimumSize(QSize(400, 60))
        self.annotate_param_groupbox.setMaximumSize(QSize(400, 60))
        self.horizontalLayout_4 = QHBoxLayout(self.annotate_param_groupbox)
        self.horizontalLayout_4.setSpacing(20)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(9, 9, 9, 9)
        self.kernel_label = QLabel(self.annotate_param_groupbox)
        self.kernel_label.setObjectName(u"kernel_label")
        sizePolicy4 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.kernel_label.sizePolicy().hasHeightForWidth())
        self.kernel_label.setSizePolicy(sizePolicy4)
        self.kernel_label.setMaximumSize(QSize(400, 25))

        self.horizontalLayout_4.addWidget(self.kernel_label, 0, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.opening_kernel_radius_input = QSpinBox(self.annotate_param_groupbox)
        self.opening_kernel_radius_input.setObjectName(u"opening_kernel_radius_input")
        sizePolicy5 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.opening_kernel_radius_input.sizePolicy().hasHeightForWidth())
        self.opening_kernel_radius_input.setSizePolicy(sizePolicy5)
        self.opening_kernel_radius_input.setMinimumSize(QSize(60, 25))
        self.opening_kernel_radius_input.setMaximumSize(QSize(60, 25))
        self.opening_kernel_radius_input.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)
        self.opening_kernel_radius_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.PlusMinus)
        self.opening_kernel_radius_input.setMaximum(20)
        self.opening_kernel_radius_input.setValue(5)

        self.horizontalLayout_4.addWidget(self.opening_kernel_radius_input, 0, Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)


        self.annotate_vertical_layout.addWidget(self.annotate_param_groupbox, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignBottom)

        self.annotate_layer_predict_groupbox = QGroupBox(self.verticalLayoutWidget_2)
        self.annotate_layer_predict_groupbox.setObjectName(u"annotate_layer_predict_groupbox")
        sizePolicy2.setHeightForWidth(self.annotate_layer_predict_groupbox.sizePolicy().hasHeightForWidth())
        self.annotate_layer_predict_groupbox.setSizePolicy(sizePolicy2)
        self.annotate_layer_predict_groupbox.setMinimumSize(QSize(400, 60))
        self.annotate_layer_predict_groupbox.setMaximumSize(QSize(400, 60))
        self.horizontalLayout_7 = QHBoxLayout(self.annotate_layer_predict_groupbox)
        self.horizontalLayout_7.setSpacing(20)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalLayout_7.setContentsMargins(9, 9, 9, 9)
        self.batch_predict_progressbar = QProgressBar(self.annotate_layer_predict_groupbox)
        self.batch_predict_progressbar.setObjectName(u"batch_predict_progressbar")
        sizePolicy6 = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        sizePolicy6.setHorizontalStretch(0)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.batch_predict_progressbar.sizePolicy().hasHeightForWidth())
        self.batch_predict_progressbar.setSizePolicy(sizePolicy6)
        self.batch_predict_progressbar.setMinimumSize(QSize(200, 25))
        self.batch_predict_progressbar.setMaximumSize(QSize(255, 25))
        self.batch_predict_progressbar.setMaximum(20)
        self.batch_predict_progressbar.setValue(0)

        self.horizontalLayout_7.addWidget(self.batch_predict_progressbar)

        self.predict_next_batch_btn = QPushButton(self.annotate_layer_predict_groupbox)
        self.predict_next_batch_btn.setObjectName(u"predict_next_batch_btn")
        self.predict_next_batch_btn.setEnabled(False)
        self.predict_next_batch_btn.setMinimumSize(QSize(0, 25))
        self.predict_next_batch_btn.setMaximumSize(QSize(250, 25))

        self.horizontalLayout_7.addWidget(self.predict_next_batch_btn, 0, Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)


        self.annotate_vertical_layout.addWidget(self.annotate_layer_predict_groupbox, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignBottom)

        self.annotate_layer_save_groupbox = QGroupBox(self.verticalLayoutWidget_2)
        self.annotate_layer_save_groupbox.setObjectName(u"annotate_layer_save_groupbox")
        sizePolicy2.setHeightForWidth(self.annotate_layer_save_groupbox.sizePolicy().hasHeightForWidth())
        self.annotate_layer_save_groupbox.setSizePolicy(sizePolicy2)
        self.annotate_layer_save_groupbox.setMinimumSize(QSize(400, 60))
        self.annotate_layer_save_groupbox.setMaximumSize(QSize(400, 60))
        self.horizontalLayout_9 = QHBoxLayout(self.annotate_layer_save_groupbox)
        self.horizontalLayout_9.setSpacing(20)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalLayout_9.setContentsMargins(9, 9, 9, 9)
        self.batch_save_progressbar = QProgressBar(self.annotate_layer_save_groupbox)
        self.batch_save_progressbar.setObjectName(u"batch_save_progressbar")
        self.batch_save_progressbar.setEnabled(False)
        sizePolicy6.setHeightForWidth(self.batch_save_progressbar.sizePolicy().hasHeightForWidth())
        self.batch_save_progressbar.setSizePolicy(sizePolicy6)
        self.batch_save_progressbar.setMinimumSize(QSize(200, 25))
        self.batch_save_progressbar.setMaximumSize(QSize(255, 25))
        self.batch_save_progressbar.setMaximum(20)
        self.batch_save_progressbar.setValue(0)

        self.horizontalLayout_9.addWidget(self.batch_save_progressbar)

        self.export_annotations_label = QLabel(self.annotate_layer_save_groupbox)
        self.export_annotations_label.setObjectName(u"export_annotations_label")
        self.export_annotations_label.setEnabled(False)

        self.horizontalLayout_9.addWidget(self.export_annotations_label, 0, Qt.AlignmentFlag.AlignVCenter)

        self.export_annotations_btn = QPushButton(self.annotate_layer_save_groupbox)
        self.export_annotations_btn.setObjectName(u"export_annotations_btn")
        self.export_annotations_btn.setEnabled(False)
        self.export_annotations_btn.setMinimumSize(QSize(0, 25))
        self.export_annotations_btn.setMaximumSize(QSize(250, 25))

        self.horizontalLayout_9.addWidget(self.export_annotations_btn, 0, Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)


        self.annotate_vertical_layout.addWidget(self.annotate_layer_save_groupbox, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignBottom)

        icon1 = QIcon()
        icon1.addFile(u"qt_gui/icons/noun-copywriting-7158879.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.toolBox.addItem(self.annotate_tab, icon1, u"Generate training data (annotate)")
        self.train_tab = QWidget()
        self.train_tab.setObjectName(u"train_tab")
        self.train_tab.setGeometry(QRect(0, 0, 410, 387))
        sizePolicy1.setHeightForWidth(self.train_tab.sizePolicy().hasHeightForWidth())
        self.train_tab.setSizePolicy(sizePolicy1)
        icon2 = QIcon()
        icon2.addFile(u"qt_gui/icons/noun-rocket-7158872.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.toolBox.addItem(self.train_tab, icon2, u"Train model")
        self.predict_tab = QWidget()
        self.predict_tab.setObjectName(u"predict_tab")
        self.predict_tab.setGeometry(QRect(0, 0, 410, 387))
        sizePolicy1.setHeightForWidth(self.predict_tab.sizePolicy().hasHeightForWidth())
        self.predict_tab.setSizePolicy(sizePolicy1)
        icon3 = QIcon()
        icon3.addFile(u"qt_gui/icons/noun-conversion-7158876.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.toolBox.addItem(self.predict_tab, icon3, u"Analyze (new) videos")

        self.mainLayout.addWidget(self.toolBox)

        self.toolBox.raise_()
        self.octron_logo.raise_()

        self.retranslateUi()

        self.toolBox.setCurrentIndex(1)
        self.toolBox.layout().setSpacing(10)


    # setupUi

    def retranslateUi(self):
        self.setWindowTitle(QCoreApplication.translate("self", u"octron_gui", None))
        self.octron_logo.setText("")
        self.toolBox.setItemText(self.toolBox.indexOf(self.project_tab), QCoreApplication.translate("self", u"Project", None))
#if QT_CONFIG(tooltip)
        self.toolBox.setItemToolTip(self.toolBox.indexOf(self.project_tab), QCoreApplication.translate("self", u"Create new octron projects or load existing ones", None))
#endif // QT_CONFIG(tooltip)
        self.horizontalGroupBox.setTitle(QCoreApplication.translate("self", u"Model selection", None))
        self.sam2model_list.setItemText(0, QCoreApplication.translate("self", u"Choose model ...", None))

        self.load_model_btn.setText(QCoreApplication.translate("self", u"Load model", None))
        self.annotate_layer_create_groupbox.setTitle(QCoreApplication.translate("self", u"Layer controls", None))
        self.label_list_combobox.setItemText(0, QCoreApplication.translate("self", u"Label ... ", None))
        self.label_list_combobox.setItemText(1, QCoreApplication.translate("self", u"\u2295 Create", None))
        self.label_list_combobox.setItemText(2, QCoreApplication.translate("self", u"\u2296 Remove", None))

#if QT_CONFIG(tooltip)
        self.label_list_combobox.setToolTip(QCoreApplication.translate("self", u"Select, add or remove labels", None))
#endif // QT_CONFIG(tooltip)
        self.label_list_combobox.setCurrentText(QCoreApplication.translate("self", u"Label ... ", None))
        self.layer_type_combobox.setItemText(0, QCoreApplication.translate("self", u"Type ... ", None))
        self.layer_type_combobox.setItemText(1, QCoreApplication.translate("self", u"Shapes", None))
        self.layer_type_combobox.setItemText(2, QCoreApplication.translate("self", u"Points", None))
        self.layer_type_combobox.setItemText(3, QCoreApplication.translate("self", u"Anchors", None))

        self.layer_type_combobox.setCurrentText(QCoreApplication.translate("self", u"Type ... ", None))
        self.create_annotation_layer_btn.setText(QCoreApplication.translate("self", u"\u2295 Create", None))
#if QT_CONFIG(tooltip)
        self.label_suffix_lineedit.setToolTip(QCoreApplication.translate("self", u"The suffix disambiguates label layers from each other\n"
"that have the same label name.\n"
"For example:\n"
"The label could be octo and suffix 1 for the first octopus,\n"
"and octo and suffix 2 for the second octo ", None))
#endif // QT_CONFIG(tooltip)
        self.label_suffix_lineedit.setPlaceholderText(QCoreApplication.translate("self", u"Suffix", None))
#if QT_CONFIG(tooltip)
        self.hard_reset_layer_btn.setToolTip(QCoreApplication.translate("self", u"Hard reset of the SAM2 predictor. Use this if prediction really did not go well for your data.", None))
#endif // QT_CONFIG(tooltip)
        self.hard_reset_layer_btn.setText(QCoreApplication.translate("self", u"\u3004 Reset", None))
#if QT_CONFIG(tooltip)
        self.create_projection_layer_btn.setToolTip(QCoreApplication.translate("self", u"Create an average projection out of all segmented images for the current label", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(statustip)
        self.create_projection_layer_btn.setStatusTip("")
#endif // QT_CONFIG(statustip)
        self.create_projection_layer_btn.setText(QCoreApplication.translate("self", u"Visualize all", None))
        self.annotate_param_groupbox.setTitle(QCoreApplication.translate("self", u"Parameters", None))
        self.kernel_label.setText(QCoreApplication.translate("self", u"Opening kernel radius", None))
        self.annotate_layer_predict_groupbox.setTitle(QCoreApplication.translate("self", u"Batch prediction", None))
#if QT_CONFIG(tooltip)
        self.batch_predict_progressbar.setToolTip(QCoreApplication.translate("self", u"<html><head/><body><p>Batch predict progress bar</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.batch_predict_progressbar.setFormat(QCoreApplication.translate("self", u"%p%", None))
        self.predict_next_batch_btn.setText("")
        self.annotate_layer_save_groupbox.setTitle(QCoreApplication.translate("self", u"Export training data", None))
#if QT_CONFIG(tooltip)
        self.batch_save_progressbar.setToolTip(QCoreApplication.translate("self", u"<html><head/><body><p>Batch predict progress bar</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.batch_save_progressbar.setFormat(QCoreApplication.translate("self", u"%p%", None))
        self.export_annotations_label.setText(QCoreApplication.translate("self", u"label name", None))
        self.export_annotations_btn.setText(QCoreApplication.translate("self", u"Export\n"
"", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.annotate_tab), QCoreApplication.translate("self", u"Generate training data (annotate)", None))
#if QT_CONFIG(tooltip)
        self.toolBox.setItemToolTip(self.toolBox.indexOf(self.annotate_tab), QCoreApplication.translate("self", u"Create annotation data for training, i.e. add segmentation or keypoint data on videos.", None))
#endif // QT_CONFIG(tooltip)
        self.toolBox.setItemText(self.toolBox.indexOf(self.train_tab), QCoreApplication.translate("self", u"Train model", None))
#if QT_CONFIG(tooltip)
        self.toolBox.setItemToolTip(self.toolBox.indexOf(self.train_tab), QCoreApplication.translate("self", u"Train a new or existing model with generated training data", None))
#endif // QT_CONFIG(tooltip)
        self.toolBox.setItemText(self.toolBox.indexOf(self.predict_tab), QCoreApplication.translate("self", u"Analyze (new) videos", None))
#if QT_CONFIG(tooltip)
        self.toolBox.setItemToolTip(self.toolBox.indexOf(self.predict_tab), QCoreApplication.translate("self", u"Use trained models to run predictions on new videos", None))
#endif // QT_CONFIG(tooltip)
    # retranslateUi





if __name__ == "__main__":
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(octron_widget(viewer))
    napari.run()