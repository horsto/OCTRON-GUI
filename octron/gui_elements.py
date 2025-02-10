# All main GUI elements

from qtpy.QtCore import QSize, QRect, Qt, QCoreApplication, Signal
from qtpy.QtGui import QCursor, QPixmap, QIcon, QPalette, QColor, QDragEnterEvent, QDropEvent
from qtpy.QtWidgets import (
    QWidget,
    QFrame,
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


class Mp4DropWidget(QWidget):
    # Signal emitted when one or more mp4 files are dropped; sends list of file paths.
    fileDropped = Signal(list)
    
    def __init__(self, parent=None, callback=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.callback = callback
        layout = QVBoxLayout(self)
        self.label = QLabel("🎥 Drag and drop .MP4 files here", self)
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)
        self.setLayout(layout)
        self._setup_styles()

    def _setup_styles(self):
        self.setAutoFillBackground(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            # Accept the event if any file URL ends with .mp4 (case insensitive)
            for url in urls:
                if url.isLocalFile() and url.toLocalFile().lower().endswith('.mp4'):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        files = []
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    if file_path.lower().endswith('.mp4'):
                        files.append(file_path)
        if files:
            # Emit signal or call callback
            self.fileDropped.emit(files)
            if self.callback is not None:
                self.callback(files)
        event.acceptProposedAction()
        
        

class octron_gui_elements(QWidget):
    '''
    Callback for octron and SAM2.
    '''
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        # Store the reference to the main OCTRON widget
        self.octron = parent
        
    ###### GUI SETUP CODE FROM QT DESIGNER ############################################################
    
    def setupUi(self, base_path):
        if not self.octron.objectName():
            self.octron.setObjectName(u"OCTRON")
        self.octron.setEnabled(True)
        self.octron.resize(410, 600)
        self.octron.setMinimumSize(QSize(410, 600))
        self.octron.setMaximumSize(QSize(410, 600))
        self.octron.setCursor(QCursor(Qt.ArrowCursor))
        self.octron.setWindowOpacity(1.000000000000000)
        self.octron.verticalLayoutWidget = QWidget(self)
        self.octron.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.octron.verticalLayoutWidget.setGeometry(QRect(0, 0, 412, 591))
        self.octron.mainLayout = QVBoxLayout(self.octron.verticalLayoutWidget)
        self.octron.mainLayout.setSpacing(10)
        self.octron.mainLayout.setObjectName(u"mainLayout")
        self.octron.mainLayout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        self.octron.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.octron.octron_logo = QLabel(self.octron.verticalLayoutWidget)
        self.octron.octron_logo.setObjectName(u"octron_logo")
        self.octron.octron_logo.setEnabled(True)
        self.octron.octron_logo.setMinimumSize(QSize(410, 120))
        self.octron.octron_logo.setMaximumSize(QSize(410, 120))
        self.octron.octron_logo.setBaseSize(QSize(0, 0))
        self.octron.octron_logo.setLineWidth(0)
        self.octron.octron_logo.setPixmap(QPixmap(f"{base_path}/qt_gui/octron_logo.svg"))
        self.octron.octron_logo.setAlignment(Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop)

        self.octron.mainLayout.addWidget(self.octron.octron_logo, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop)

        self.octron.toolBox = QToolBox(self.octron.verticalLayoutWidget)
        self.octron.toolBox.setObjectName(u"toolBox")
        self.octron.toolBox.setEnabled(True)
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.octron.toolBox.sizePolicy().hasHeightForWidth())
        self.octron.toolBox.setSizePolicy(sizePolicy)
        self.octron.toolBox.setMinimumSize(QSize(410, 450))
        self.octron.toolBox.setMaximumSize(QSize(410, 450))
        self.octron.toolBox.setCursor(QCursor(Qt.ArrowCursor))
        self.octron.toolBox.setFrameShape(QFrame.Shape.NoFrame)
        self.octron.toolBox.setFrameShadow(QFrame.Shadow.Plain)
        self.octron.toolBox.setLineWidth(0)
        self.octron.toolBox.setMidLineWidth(0)
        self.octron.project_tab = QWidget()
        self.octron.project_tab.setObjectName(u"project_tab")
        self.octron.project_tab.setGeometry(QRect(0, 0, 410, 314))
        sizePolicy1 = QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.octron.project_tab.sizePolicy().hasHeightForWidth())
        self.octron.project_tab.setSizePolicy(sizePolicy1)
        self.octron.verticalLayoutWidget_3 = QWidget(self.octron.project_tab)
        self.octron.verticalLayoutWidget_3.setObjectName(u"verticalLayoutWidget_3")
        self.octron.verticalLayoutWidget_3.setGeometry(QRect(0, -1, 402, 241))
        self.octron.project_vertical_layout = QVBoxLayout(self.octron.verticalLayoutWidget_3)
        self.octron.project_vertical_layout.setObjectName(u"project_vertical_layout")
        self.octron.project_vertical_layout.setContentsMargins(0, 0, 0, 0)
        self.octron.folder_sect_groupbox = QGroupBox(self.octron.verticalLayoutWidget_3)
        self.octron.folder_sect_groupbox.setObjectName(u"folder_sect_groupbox")
        sizePolicy2 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.octron.folder_sect_groupbox.sizePolicy().hasHeightForWidth())
        self.octron.folder_sect_groupbox.setSizePolicy(sizePolicy2)
        self.octron.folder_sect_groupbox.setMinimumSize(QSize(400, 50))
        self.octron.folder_sect_groupbox.setMaximumSize(QSize(400, 50))
        self.octron.horizontalLayout_11 = QHBoxLayout(self.octron.folder_sect_groupbox)
        self.octron.horizontalLayout_11.setSpacing(20)
        self.octron.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.octron.horizontalLayout_11.setContentsMargins(9, 9, 9, 9)
        self.octron.create_project_btn = QPushButton(self.octron.folder_sect_groupbox)
        self.octron.create_project_btn.setObjectName(u"create_project_btn")
        self.octron.create_project_btn.setMinimumSize(QSize(250, 25))
        self.octron.create_project_btn.setMaximumSize(QSize(300, 25))

        self.octron.horizontalLayout_11.addWidget(self.octron.create_project_btn, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignVCenter)


        self.octron.project_vertical_layout.addWidget(self.octron.folder_sect_groupbox, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop)

        self.octron.project_video_drop_groupbox = QGroupBox(self.octron.verticalLayoutWidget_3)
        self.octron.project_video_drop_groupbox.setObjectName(u"project_video_drop_groupbox")
        sizePolicy2.setHeightForWidth(self.octron.project_video_drop_groupbox.sizePolicy().hasHeightForWidth())
        self.octron.project_video_drop_groupbox.setSizePolicy(sizePolicy2)
        self.octron.project_video_drop_groupbox.setMinimumSize(QSize(400, 150))
        self.octron.project_video_drop_groupbox.setMaximumSize(QSize(400, 150))
        self.octron.horizontalLayout = QHBoxLayout(self.octron.project_video_drop_groupbox)
        self.octron.horizontalLayout.setSpacing(20)
        self.octron.horizontalLayout.setObjectName(u"horizontalLayout")
        self.octron.horizontalLayout.setContentsMargins(9, 9, 9, 9)
        self.octron.video_file_drop_widget = Mp4DropWidget(callback=self.octron.on_file_dropped_area)
        self.octron.video_file_drop_widget.setObjectName(u"video_file_drop_widget")
        self.octron.video_file_drop_widget.setMinimumSize(QSize(380, 110))
        self.octron.video_file_drop_widget.setMaximumSize(QSize(380, 110))

        self.octron.horizontalLayout.addWidget(self.octron.video_file_drop_widget)


        self.octron.project_vertical_layout.addWidget(self.octron.project_video_drop_groupbox, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop)

        icon = QIcon()
        icon.addFile(f"{base_path}/qt_gui/icons/noun-project-7158867.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.octron.toolBox.addItem(self.octron.project_tab, icon, u"Project")
        self.octron.annotate_tab = QWidget()
        self.octron.annotate_tab.setObjectName(u"annotate_tab")
        self.octron.annotate_tab.setGeometry(QRect(0, 0, 405, 314))
        sizePolicy1.setHeightForWidth(self.octron.annotate_tab.sizePolicy().hasHeightForWidth())
        self.octron.annotate_tab.setSizePolicy(sizePolicy1)
        self.octron.annotate_tab.setMaximumSize(QSize(405, 700))
        self.octron.verticalLayoutWidget_2 = QWidget(self.octron.annotate_tab)
        self.octron.verticalLayoutWidget_2.setObjectName(u"verticalLayoutWidget_2")
        self.octron.verticalLayoutWidget_2.setGeometry(QRect(0, 0, 402, 391))
        self.octron.annotate_vertical_layout = QVBoxLayout(self.octron.verticalLayoutWidget_2)
#ifndef Q_OS_MAC
        self.octron.annotate_vertical_layout.setSpacing(-1)
#endif
        self.octron.annotate_vertical_layout.setObjectName(u"annotate_vertical_layout")
        self.octron.annotate_vertical_layout.setContentsMargins(0, 0, 0, 10)
        self.octron.horizontalGroupBox = QGroupBox(self.octron.verticalLayoutWidget_2)
        self.octron.horizontalGroupBox.setObjectName(u"horizontalGroupBox")
        sizePolicy2.setHeightForWidth(self.octron.horizontalGroupBox.sizePolicy().hasHeightForWidth())
        self.octron.horizontalGroupBox.setSizePolicy(sizePolicy2)
        self.octron.horizontalGroupBox.setMinimumSize(QSize(400, 60))
        self.octron.horizontalGroupBox.setMaximumSize(QSize(400, 60))
        self.octron.horizontalLayout_8 = QHBoxLayout(self.octron.horizontalGroupBox)
        self.octron.horizontalLayout_8.setSpacing(20)
        self.octron.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.octron.horizontalLayout_8.setContentsMargins(9, 9, 9, 9)
        self.octron.sam2model_list = QComboBox(self.octron.horizontalGroupBox)
        self.octron.sam2model_list.addItem("")
        self.octron.sam2model_list.setObjectName(u"sam2model_list")
        self.octron.sam2model_list.setMinimumSize(QSize(167, 25))
        self.octron.sam2model_list.setMaximumSize(QSize(167, 25))

        self.octron.horizontalLayout_8.addWidget(self.octron.sam2model_list, 0, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.octron.load_model_btn = QPushButton(self.octron.horizontalGroupBox)
        self.octron.load_model_btn.setObjectName(u"load_model_btn")
        self.octron.load_model_btn.setMinimumSize(QSize(0, 25))
        self.octron.load_model_btn.setMaximumSize(QSize(250, 25))

        self.octron.horizontalLayout_8.addWidget(self.octron.load_model_btn, 0, Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)


        self.octron.annotate_vertical_layout.addWidget(self.octron.horizontalGroupBox, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop)

        self.octron.annotate_layer_create_groupbox = QGroupBox(self.octron.verticalLayoutWidget_2)
        self.octron.annotate_layer_create_groupbox.setObjectName(u"annotate_layer_create_groupbox")
        sizePolicy2.setHeightForWidth(self.octron.annotate_layer_create_groupbox.sizePolicy().hasHeightForWidth())
        self.octron.annotate_layer_create_groupbox.setSizePolicy(sizePolicy2)
        self.octron.annotate_layer_create_groupbox.setMinimumSize(QSize(400, 90))
        self.octron.annotate_layer_create_groupbox.setMaximumSize(QSize(400, 90))
        self.octron.gridLayout = QGridLayout(self.octron.annotate_layer_create_groupbox)
        self.octron.gridLayout.setObjectName(u"gridLayout")
        self.octron.gridLayout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.octron.gridLayout.setHorizontalSpacing(20)
        self.octron.gridLayout.setVerticalSpacing(9)
        self.octron.gridLayout.setContentsMargins(9, 9, 9, 9)
        self.octron.label_list_combobox = QComboBox(self.octron.annotate_layer_create_groupbox)
        self.octron.label_list_combobox.addItem("")
        self.octron.label_list_combobox.addItem("")
        self.octron.label_list_combobox.addItem("")
        self.octron.label_list_combobox.setObjectName(u"label_list_combobox")
        self.octron.label_list_combobox.setMinimumSize(QSize(110, 25))
        self.octron.label_list_combobox.setMaximumSize(QSize(110, 25))
        self.octron.label_list_combobox.setEditable(False)
        self.octron.label_list_combobox.setMaxCount(15)
        self.octron.label_list_combobox.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.octron.label_list_combobox.setIconSize(QSize(14, 14))
        self.octron.label_list_combobox.setFrame(False)

        self.octron.gridLayout.addWidget(self.octron.label_list_combobox, 0, 2, 1, 1, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.octron.layer_type_combobox = QComboBox(self.octron.annotate_layer_create_groupbox)
        self.octron.layer_type_combobox.addItem("")
        self.octron.layer_type_combobox.addItem("")
        self.octron.layer_type_combobox.addItem("")
        self.octron.layer_type_combobox.addItem("")
        self.octron.layer_type_combobox.setObjectName(u"layer_type_combobox")
        self.octron.layer_type_combobox.setMinimumSize(QSize(110, 25))
        self.octron.layer_type_combobox.setMaximumSize(QSize(110, 25))
        self.octron.layer_type_combobox.setMaxCount(15)
        self.octron.layer_type_combobox.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.octron.layer_type_combobox.setIconSize(QSize(14, 14))
        self.octron.layer_type_combobox.setFrame(False)

        self.octron.gridLayout.addWidget(self.octron.layer_type_combobox, 0, 0, 1, 1, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.octron.create_annotation_layer_btn = QPushButton(self.octron.annotate_layer_create_groupbox)
        self.octron.create_annotation_layer_btn.setObjectName(u"create_annotation_layer_btn")
        self.octron.create_annotation_layer_btn.setMinimumSize(QSize(70, 25))
        self.octron.create_annotation_layer_btn.setMaximumSize(QSize(70, 25))

        self.octron.gridLayout.addWidget(self.octron.create_annotation_layer_btn, 0, 4, 1, 1, Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)

        self.octron.label_suffix_lineedit = QLineEdit(self.octron.annotate_layer_create_groupbox)
        self.octron.label_suffix_lineedit.setObjectName(u"label_suffix_lineedit")
        self.octron.label_suffix_lineedit.setMinimumSize(QSize(60, 25))
        self.octron.label_suffix_lineedit.setMaximumSize(QSize(60, 25))
        self.octron.label_suffix_lineedit.setInputMask(u"")
        self.octron.label_suffix_lineedit.setText(u"")
        self.octron.label_suffix_lineedit.setMaxLength(100)

        self.octron.gridLayout.addWidget(self.octron.label_suffix_lineedit, 0, 3, 1, 1, Qt.AlignmentFlag.AlignVCenter)

        self.octron.hard_reset_layer_btn = QPushButton(self.octron.annotate_layer_create_groupbox)
        self.octron.hard_reset_layer_btn.setObjectName(u"hard_reset_layer_btn")
        self.octron.hard_reset_layer_btn.setMinimumSize(QSize(70, 25))
        self.octron.hard_reset_layer_btn.setMaximumSize(QSize(70, 25))
        self.octron.hard_reset_layer_btn.setAutoRepeatInterval(2000)

        self.octron.gridLayout.addWidget(self.octron.hard_reset_layer_btn, 1, 4, 1, 1, Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)

        self.octron.create_projection_layer_btn = QPushButton(self.octron.annotate_layer_create_groupbox)
        self.octron.create_projection_layer_btn.setObjectName(u"create_projection_layer_btn")
        self.octron.create_projection_layer_btn.setMinimumSize(QSize(110, 25))
        self.octron.create_projection_layer_btn.setMaximumSize(QSize(110, 25))

        self.octron.gridLayout.addWidget(self.octron.create_projection_layer_btn, 1, 0, 1, 1, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)


        self.octron.annotate_vertical_layout.addWidget(self.octron.annotate_layer_create_groupbox, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop)

        self.octron.annotate_param_groupbox = QGroupBox(self.octron.verticalLayoutWidget_2)
        self.octron.annotate_param_groupbox.setObjectName(u"annotate_param_groupbox")
        self.octron.annotate_param_groupbox.setEnabled(False)
        sizePolicy3 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        sizePolicy3.setHorizontalStretch(100)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.octron.annotate_param_groupbox.sizePolicy().hasHeightForWidth())
        self.octron.annotate_param_groupbox.setSizePolicy(sizePolicy3)
        self.octron.annotate_param_groupbox.setMinimumSize(QSize(400, 60))
        self.octron.annotate_param_groupbox.setMaximumSize(QSize(400, 60))
        self.octron.horizontalLayout_4 = QHBoxLayout(self.octron.annotate_param_groupbox)
        self.octron.horizontalLayout_4.setSpacing(20)
        self.octron.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.octron.horizontalLayout_4.setContentsMargins(9, 9, 9, 9)
        self.octron.kernel_label = QLabel(self.octron.annotate_param_groupbox)
        self.octron.kernel_label.setObjectName(u"kernel_label")
        sizePolicy4 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.octron.kernel_label.sizePolicy().hasHeightForWidth())
        self.octron.kernel_label.setSizePolicy(sizePolicy4)
        self.octron.kernel_label.setMaximumSize(QSize(400, 25))

        self.octron.horizontalLayout_4.addWidget(self.octron.kernel_label, 0, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.octron.opening_kernel_radius_input = QSpinBox(self.octron.annotate_param_groupbox)
        self.octron.opening_kernel_radius_input.setObjectName(u"opening_kernel_radius_input")
        sizePolicy5 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.octron.opening_kernel_radius_input.sizePolicy().hasHeightForWidth())
        self.octron.opening_kernel_radius_input.setSizePolicy(sizePolicy5)
        self.octron.opening_kernel_radius_input.setMinimumSize(QSize(60, 25))
        self.octron.opening_kernel_radius_input.setMaximumSize(QSize(60, 25))
        self.octron.opening_kernel_radius_input.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)
        self.octron.opening_kernel_radius_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.PlusMinus)
        self.octron.opening_kernel_radius_input.setMaximum(20)
        self.octron.opening_kernel_radius_input.setValue(5)

        self.octron.horizontalLayout_4.addWidget(self.octron.opening_kernel_radius_input, 0, Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)


        self.octron.annotate_vertical_layout.addWidget(self.octron.annotate_param_groupbox, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignBottom)

        self.octron.annotate_layer_predict_groupbox = QGroupBox(self.octron.verticalLayoutWidget_2)
        self.octron.annotate_layer_predict_groupbox.setObjectName(u"annotate_layer_predict_groupbox")
        sizePolicy2.setHeightForWidth(self.octron.annotate_layer_predict_groupbox.sizePolicy().hasHeightForWidth())
        self.octron.annotate_layer_predict_groupbox.setSizePolicy(sizePolicy2)
        self.octron.annotate_layer_predict_groupbox.setMinimumSize(QSize(400, 60))
        self.octron.annotate_layer_predict_groupbox.setMaximumSize(QSize(400, 60))
        self.octron.horizontalLayout_7 = QHBoxLayout(self.octron.annotate_layer_predict_groupbox)
        self.octron.horizontalLayout_7.setSpacing(20)
        self.octron.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.octron.horizontalLayout_7.setContentsMargins(9, 9, 9, 9)
        self.octron.batch_predict_progressbar = QProgressBar(self.octron.annotate_layer_predict_groupbox)
        self.octron.batch_predict_progressbar.setObjectName(u"batch_predict_progressbar")
        sizePolicy6 = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        sizePolicy6.setHorizontalStretch(0)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.octron.batch_predict_progressbar.sizePolicy().hasHeightForWidth())
        self.octron.batch_predict_progressbar.setSizePolicy(sizePolicy6)
        self.octron.batch_predict_progressbar.setMinimumSize(QSize(200, 25))
        self.octron.batch_predict_progressbar.setMaximumSize(QSize(255, 25))
        self.octron.batch_predict_progressbar.setMaximum(20)
        self.octron.batch_predict_progressbar.setValue(0)

        self.octron.horizontalLayout_7.addWidget(self.octron.batch_predict_progressbar)

        self.octron.predict_next_batch_btn = QPushButton(self.octron.annotate_layer_predict_groupbox)
        self.octron.predict_next_batch_btn.setObjectName(u"predict_next_batch_btn")
        self.octron.predict_next_batch_btn.setEnabled(False)
        self.octron.predict_next_batch_btn.setMinimumSize(QSize(0, 25))
        self.octron.predict_next_batch_btn.setMaximumSize(QSize(250, 25))

        self.octron.horizontalLayout_7.addWidget(self.octron.predict_next_batch_btn, 0, Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)


        self.octron.annotate_vertical_layout.addWidget(self.octron.annotate_layer_predict_groupbox, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignBottom)

        self.octron.annotate_layer_save_groupbox = QGroupBox(self.octron.verticalLayoutWidget_2)
        self.octron.annotate_layer_save_groupbox.setObjectName(u"annotate_layer_save_groupbox")
        sizePolicy2.setHeightForWidth(self.octron.annotate_layer_save_groupbox.sizePolicy().hasHeightForWidth())
        self.octron.annotate_layer_save_groupbox.setSizePolicy(sizePolicy2)
        self.octron.annotate_layer_save_groupbox.setMinimumSize(QSize(400, 60))
        self.octron.annotate_layer_save_groupbox.setMaximumSize(QSize(400, 60))
        self.octron.horizontalLayout_9 = QHBoxLayout(self.octron.annotate_layer_save_groupbox)
        self.octron.horizontalLayout_9.setSpacing(20)
        self.octron.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.octron.horizontalLayout_9.setContentsMargins(9, 9, 9, 9)
        self.octron.batch_save_progressbar = QProgressBar(self.octron.annotate_layer_save_groupbox)
        self.octron.batch_save_progressbar.setObjectName(u"batch_save_progressbar")
        self.octron.batch_save_progressbar.setEnabled(False)
        sizePolicy6.setHeightForWidth(self.octron.batch_save_progressbar.sizePolicy().hasHeightForWidth())
        self.octron.batch_save_progressbar.setSizePolicy(sizePolicy6)
        self.octron.batch_save_progressbar.setMinimumSize(QSize(200, 25))
        self.octron.batch_save_progressbar.setMaximumSize(QSize(255, 25))
        self.octron.batch_save_progressbar.setMaximum(20)
        self.octron.batch_save_progressbar.setValue(0)

        self.octron.horizontalLayout_9.addWidget(self.octron.batch_save_progressbar)

        self.octron.export_annotations_label = QLabel(self.octron.annotate_layer_save_groupbox)
        self.octron.export_annotations_label.setObjectName(u"export_annotations_label")
        self.octron.export_annotations_label.setEnabled(False)

        self.octron.horizontalLayout_9.addWidget(self.octron.export_annotations_label, 0, Qt.AlignmentFlag.AlignVCenter)

        self.octron.export_annotations_btn = QPushButton(self.octron.annotate_layer_save_groupbox)
        self.octron.export_annotations_btn.setObjectName(u"export_annotations_btn")
        self.octron.export_annotations_btn.setEnabled(False)
        self.octron.export_annotations_btn.setMinimumSize(QSize(0, 25))
        self.octron.export_annotations_btn.setMaximumSize(QSize(250, 25))

        self.octron.horizontalLayout_9.addWidget(self.octron.export_annotations_btn, 0, Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)


        self.octron.annotate_vertical_layout.addWidget(self.octron.annotate_layer_save_groupbox, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignBottom)

        icon1 = QIcon()
        icon1.addFile(f"{base_path}/qt_gui/icons/noun-copywriting-7158879.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.octron.toolBox.addItem(self.octron.annotate_tab, icon1, u"Generate training data (annotate)")
        self.octron.train_tab = QWidget()
        self.octron.train_tab.setObjectName(u"train_tab")
        self.octron.train_tab.setGeometry(QRect(0, 0, 410, 314))
        sizePolicy1.setHeightForWidth(self.octron.train_tab.sizePolicy().hasHeightForWidth())
        self.octron.train_tab.setSizePolicy(sizePolicy1)
        icon2 = QIcon()
        icon2.addFile(f"{base_path}/qt_gui/icons/noun-rocket-7158872.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.octron.toolBox.addItem(self.octron.train_tab, icon2, u"Train model")
        self.octron.predict_tab = QWidget()
        self.octron.predict_tab.setObjectName(u"predict_tab")
        self.octron.predict_tab.setGeometry(QRect(0, 0, 410, 314))
        sizePolicy1.setHeightForWidth(self.octron.predict_tab.sizePolicy().hasHeightForWidth())
        self.octron.predict_tab.setSizePolicy(sizePolicy1)
        icon3 = QIcon()
        icon3.addFile(f"{base_path}/qt_gui/icons/noun-conversion-7158876.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.octron.toolBox.addItem(self.octron.predict_tab, icon3, u"Analyze (new) videos")

        self.octron.mainLayout.addWidget(self.octron.toolBox, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop)

        self.octron.toolBox.raise_()
        self.octron.octron_logo.raise_()

        self.octron.toolBox.setCurrentIndex(0)
        self.octron.toolBox.layout().setSpacing(10)


    # setupUi
        self.octron.setWindowTitle(QCoreApplication.translate("self", u"octron_gui", None))
        self.octron.octron_logo.setText("")
        self.octron.folder_sect_groupbox.setTitle("")
        self.octron.create_project_btn.setText(QCoreApplication.translate("self", u"\u2295 Create octron project", None))
        self.octron.project_video_drop_groupbox.setTitle(QCoreApplication.translate("self", u"Video files", None))
#if QT_CONFIG(tooltip)
        self.octron.video_file_drop_widget.setToolTip(QCoreApplication.translate("self", u"Drag and drop .mp4 files here", None))
#endif // QT_CONFIG(tooltip)
        self.octron.toolBox.setItemText(self.octron.toolBox.indexOf(self.octron.project_tab), QCoreApplication.translate("self", u"Project", None))
#if QT_CONFIG(tooltip)
        self.octron.toolBox.setItemToolTip(self.octron.toolBox.indexOf(self.octron.project_tab), QCoreApplication.translate("self", u"Create new octron projects or load existing ones", None))
#endif // QT_CONFIG(tooltip)
        self.octron.horizontalGroupBox.setTitle(QCoreApplication.translate("self", u"Model selection", None))
        self.octron.sam2model_list.setItemText(0, QCoreApplication.translate("self", u"Choose model ...", None))

        self.octron.load_model_btn.setText(QCoreApplication.translate("self", u"Load model", None))
        self.octron.annotate_layer_create_groupbox.setTitle(QCoreApplication.translate("self", u"Layer controls", None))
        self.octron.label_list_combobox.setItemText(0, QCoreApplication.translate("self", u"Label ... ", None))
        self.octron.label_list_combobox.setItemText(1, QCoreApplication.translate("self", u"\u2295 Create", None))
        self.octron.label_list_combobox.setItemText(2, QCoreApplication.translate("self", u"\u2296 Remove", None))

#if QT_CONFIG(tooltip)
        self.octron.label_list_combobox.setToolTip(QCoreApplication.translate("self", u"Select, add or remove labels", None))
#endif // QT_CONFIG(tooltip)
        self.octron.label_list_combobox.setCurrentText(QCoreApplication.translate("self", u"Label ... ", None))
        self.octron.layer_type_combobox.setItemText(0, QCoreApplication.translate("self", u"Type ... ", None))
        self.octron.layer_type_combobox.setItemText(1, QCoreApplication.translate("self", u"Shapes", None))
        self.octron.layer_type_combobox.setItemText(2, QCoreApplication.translate("self", u"Points", None))
        self.octron.layer_type_combobox.setItemText(3, QCoreApplication.translate("self", u"Anchors", None))

        self.octron.layer_type_combobox.setCurrentText(QCoreApplication.translate("self", u"Type ... ", None))
        self.octron.create_annotation_layer_btn.setText(QCoreApplication.translate("self", u"\u2295 Create", None))
#if QT_CONFIG(tooltip)
        self.octron.label_suffix_lineedit.setToolTip(QCoreApplication.translate("self", u"The suffix disambiguates label layers from each other\n"
"that have the same label name.\n"
"For example:\n"
"The label could be octo and suffix 1 for the first octopus,\n"
"and octo and suffix 2 for the second octo ", None))
#endif // QT_CONFIG(tooltip)
        self.octron.label_suffix_lineedit.setPlaceholderText(QCoreApplication.translate("self", u"Suffix", None))
#if QT_CONFIG(tooltip)
        self.octron.hard_reset_layer_btn.setToolTip(QCoreApplication.translate("self", u"Hard reset of the SAM2 predictor. Use this if prediction really did not go well for your data.", None))
#endif // QT_CONFIG(tooltip)
        self.octron.hard_reset_layer_btn.setText(QCoreApplication.translate("self", u"\u3004 Reset", None))
#if QT_CONFIG(tooltip)
        self.octron.create_projection_layer_btn.setToolTip(QCoreApplication.translate("self", u"Create an average projection out of all segmented images for the current label", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(statustip)
        self.octron.create_projection_layer_btn.setStatusTip("")
#endif // QT_CONFIG(statustip)
        self.octron.create_projection_layer_btn.setText(QCoreApplication.translate("self", u"Visualize all", None))
        self.octron.annotate_param_groupbox.setTitle(QCoreApplication.translate("self", u"Parameters", None))
        self.octron.kernel_label.setText(QCoreApplication.translate("self", u"Opening kernel radius", None))
        self.octron.annotate_layer_predict_groupbox.setTitle(QCoreApplication.translate("self", u"Batch prediction", None))
#if QT_CONFIG(tooltip)
        self.octron.batch_predict_progressbar.setToolTip(QCoreApplication.translate("self", u"<html><head/><body><p>Batch predict progress bar</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.octron.batch_predict_progressbar.setFormat(QCoreApplication.translate("self", u"%p%", None))
        self.octron.predict_next_batch_btn.setText("")
        self.octron.annotate_layer_save_groupbox.setTitle(QCoreApplication.translate("self", u"Export training data", None))
#if QT_CONFIG(tooltip)
        self.octron.batch_save_progressbar.setToolTip(QCoreApplication.translate("self", u"<html><head/><body><p>Batch predict progress bar</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.octron.batch_save_progressbar.setFormat(QCoreApplication.translate("self", u"%p%", None))
        self.octron.export_annotations_label.setText(QCoreApplication.translate("self", u"label name", None))
        self.octron.export_annotations_btn.setText(QCoreApplication.translate("self", u"Export\n"
"", None))
        self.octron.toolBox.setItemText(self.octron.toolBox.indexOf(self.octron.annotate_tab), QCoreApplication.translate("self", u"Generate training data (annotate)", None))
#if QT_CONFIG(tooltip)
        self.octron.toolBox.setItemToolTip(self.octron.toolBox.indexOf(self.octron.annotate_tab), QCoreApplication.translate("self", u"Create annotation data for training, i.e. add segmentation or keypoint data on videos.", None))
#endif // QT_CONFIG(tooltip)
        self.octron.toolBox.setItemText(self.octron.toolBox.indexOf(self.octron.train_tab), QCoreApplication.translate("self", u"Train model", None))
#if QT_CONFIG(tooltip)
        self.octron.toolBox.setItemToolTip(self.octron.toolBox.indexOf(self.octron.train_tab), QCoreApplication.translate("self", u"Train a new or existing model with generated training data", None))
#endif // QT_CONFIG(tooltip)
        self.octron.toolBox.setItemText(self.octron.toolBox.indexOf(self.octron.predict_tab), QCoreApplication.translate("self", u"Analyze (new) videos", None))
#if QT_CONFIG(tooltip)
        self.octron.toolBox.setItemToolTip(self.octron.toolBox.indexOf(self.octron.predict_tab), QCoreApplication.translate("self", u"Use trained models to run predictions on new videos", None))
#endif // QT_CONFIG(tooltip)
    # retranslateUi

