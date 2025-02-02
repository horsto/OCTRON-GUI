'''
OCTRON



'''
import os 
from pathlib import Path
cur_path = Path(os.path.abspath(__file__)).parent


from qtpy.QtCore import *  # type: ignore
from qtpy.QtGui import *  # type: ignore
from qtpy.QtWidgets import *  # type: ignore


import napari
import napari.window
from napari.utils.notifications import show_info, show_warning


# SAM2 specific 
import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
from octron.sam2_octron.helpers.build_sam2_octron import build_sam2_octron  


from qtpy.QtWidgets import QWidget

class octron_widget(QWidget):
    '''
    
    '''
    def __init__(self, viewer: 'napari.viewer.Viewer', parent=None):
        super().__init__(parent)
        
        
        self._viewer = viewer
        
        self.setupUi()
        
        # self.predictor, self.device = self._initialize_sam2()
    def setupUi(self):
        if not self.objectName():
            self.setObjectName(u"self")
        self.setEnabled(True)
        self.resize(410, 500)
        self.setMinimumSize(QSize(410, 500))
        self.setMaximumSize(QSize(410, 1000))
        self.setCursor(QCursor(Qt.ArrowCursor))
        self.setWindowOpacity(1.000000000000000)
        self.verticalLayoutWidget = QWidget(self)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QRect(0, 0, 412, 501))
        self.mainLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.mainLayout.setSpacing(0)
        self.mainLayout.setObjectName(u"mainLayout")
        self.mainLayout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.octron_logo = QLabel(self.verticalLayoutWidget)
        self.octron_logo.setObjectName(u"octron_logo")
        self.octron_logo.setEnabled(True)
        self.octron_logo.setMinimumSize(QSize(410, 100))
        self.octron_logo.setBaseSize(QSize(0, 0))
        self.octron_logo.setPixmap(QPixmap(u"qt_gui/octron_logo.svg"))
        self.octron_logo.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop)

        self.mainLayout.addWidget(self.octron_logo, 0, Qt.AlignmentFlag.AlignLeft)

        self.toolBox = QToolBox(self.verticalLayoutWidget)
        self.toolBox.setObjectName(u"toolBox")
        self.toolBox.setEnabled(True)
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.toolBox.sizePolicy().hasHeightForWidth())
        self.toolBox.setSizePolicy(sizePolicy)
        self.toolBox.setMaximumSize(QSize(410, 750))
        self.toolBox.setCursor(QCursor(Qt.ArrowCursor))
        self.toolBox.setFrameShape(QFrame.Shape.NoFrame)
        self.toolBox.setFrameShadow(QFrame.Shadow.Plain)
        self.toolBox.setLineWidth(1)
        self.toolBox.setMidLineWidth(1)
        self.project_tab = QWidget()
        self.project_tab.setObjectName(u"project_tab")
        self.project_tab.setGeometry(QRect(0, 0, 410, 224))
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
        self.annotate_tab.setGeometry(QRect(0, 0, 410, 224))
        sizePolicy1.setHeightForWidth(self.annotate_tab.sizePolicy().hasHeightForWidth())
        self.annotate_tab.setSizePolicy(sizePolicy1)
        icon1 = QIcon()
        icon1.addFile(u"qt_gui/icons/noun-copywriting-7158879.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.toolBox.addItem(self.annotate_tab, icon1, u"Generate training data (annotate)")
        self.train_tab = QWidget()
        self.train_tab.setObjectName(u"train_tab")
        self.train_tab.setGeometry(QRect(0, 0, 410, 224))
        sizePolicy1.setHeightForWidth(self.train_tab.sizePolicy().hasHeightForWidth())
        self.train_tab.setSizePolicy(sizePolicy1)
        icon2 = QIcon()
        icon2.addFile(u"qt_gui/icons/noun-rocket-7158872.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.toolBox.addItem(self.train_tab, icon2, u"Train model")
        self.predict_tab = QWidget()
        self.predict_tab.setObjectName(u"predict_tab")
        self.predict_tab.setGeometry(QRect(0, 0, 410, 224))
        sizePolicy1.setHeightForWidth(self.predict_tab.sizePolicy().hasHeightForWidth())
        self.predict_tab.setSizePolicy(sizePolicy1)
        icon3 = QIcon()
        icon3.addFile(u"qt_gui/icons/noun-conversion-7158876.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.toolBox.addItem(self.predict_tab, icon3, u"Analyze (new) videos")

        self.mainLayout.addWidget(self.toolBox)

        self.toolBox.raise_()
        self.octron_logo.raise_()

        self.retranslateUi()

        self.toolBox.setCurrentIndex(0)


    # setupUi

    def retranslateUi(self):
        self.setWindowTitle(QCoreApplication.translate("self", u"octron_gui", None))
        self.octron_logo.setText("")
        self.toolBox.setItemText(self.toolBox.indexOf(self.project_tab), QCoreApplication.translate("self", u"Project", None))
#if QT_CONFIG(tooltip)
        self.toolBox.setItemToolTip(self.toolBox.indexOf(self.project_tab), QCoreApplication.translate("self", u"Create new octron projects or load existing ones", None))
#endif // QT_CONFIG(tooltip)
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


     
     
     
    # def _initialize_sam2(self):
    #     '''
    #     Initialize the SAM2 model
    #     '''
    #     sam2_folder = Path('sam2_octron')
        
        
    #     # TODO: Make checkpoint path and config file path configurable  
    #     checkpoint = 'sam2.1_hiera_large.pt' # under folder /checkpoints
    #     model_cfg = 'sam2.1/sam2.1_hiera_l.yaml' # under folder /configs
    #     # ------------------------------------------------------------------------------------
    #     sam2_checkpoint = cur_path / sam2_folder / Path(f'checkpoints/{checkpoint}')
    #     model_cfg = Path(f'configs/{model_cfg}')
        
    #     assert sam2_checkpoint.exists(), f'Checkpoint file does not exist: {sam2_checkpoint}'
    #     assert (cur_path/sam2_folder/model_cfg).exists(), f'Config file does not exist: {cur_path/sam2_folder/model_cfg}'
    #     predictor, device  = build_sam2_video_predictor_octron(config_file=model_cfg.as_posix(), 
    #                                                            ckpt_path=sam2_checkpoint.as_posix(), 
    #                                                            )
                
    #     return predictor, device    
    
    # def _on_click(self):
    #     show_info('Hello and welcome to OCTRON!\nOctopuses are amazing creatures 🐙')
        
    # def _forw_1frame(self):
    #     current_indices = self._viewer.dims.current_step
    #     self._viewer.dims.set_point(0,current_indices[0]+100)
        
        
        
        
