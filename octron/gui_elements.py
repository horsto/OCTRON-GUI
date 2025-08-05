# All main GUI elements
import pathlib
from qtpy.QtCore import QSize, QRect, Qt, QCoreApplication, Signal
from qtpy.QtGui import QCursor, QPixmap, QIcon, QDragEnterEvent, QDropEvent
from qtpy.QtWidgets import (
    QWidget,
    QFrame,
    QVBoxLayout,
    QLabel,
    QLayout,
    QToolBox,
    QComboBox,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QSizePolicy,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QTableView,
    QProgressBar,
    QPushButton,
    QAbstractSpinBox,
    QGridLayout,
    QFileDialog,
    QAbstractItemView
)      

from importlib.metadata import version
__version__ = version("octron")

class Mp4DropWidget(QWidget):
    # Signal emitted when one or more mp4 files are dropped; sends list of file paths.
    fileDropped = Signal(list)
    
    def __init__(self, parent=None, callback=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.callback = callback
        layout = QVBoxLayout(self)
        self.label = QLabel("🎥 Click, or drag and drop .MP4 file here", self)
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
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            file, _ = QFileDialog.getOpenFileName(self, "Select MP4 file", "", "MP4 Files (*.mp4)")
            if file:
                self.fileDropped.emit([file])
                if self.callback is not None:
                    self.callback([file])
            event.accept()
        else:
            super().mousePressEvent(event)

class octron_gui_elements(QWidget):
    def __init__(self, parent: QWidget, base_path: pathlib.Path):
        """
        Initializes the GUI elements for the OCTRON application.
        
        Parameters
        ----------
        parent : QWidget
            The parent widget for the GUI elements.
        base_path : pathlib.Path
                The base path for loading resources (e.g., icons, images).
        
        """
        super().__init__(parent)
        self.octron = parent
        
        # Iniatialize the GUI elements
        self.setupUi(base_path)
        
        # Correct window title and app name post-hoc
        self.octron.setWindowTitle(f"OCTRON v{__version__}")
        self.octron.setObjectName(f"OCTRON v{__version__}")
        
    ###### GUI SETUP CODE FROM QT DESIGNER ############################################################
    def setupUi(self, base_path):
        if not self.octron.objectName():
            self.octron.setObjectName(u"self")
        self.octron.setEnabled(True)
        self.octron.resize(410, 700)
        self.octron.setMinimumSize(QSize(410, 700))
        self.octron.setMaximumSize(QSize(410, 700))
        self.octron.setCursor(QCursor(Qt.ArrowCursor))
        self.octron.setWindowOpacity(1.000000000000000)
        self.octron.verticalLayoutWidget = QWidget(self)
        self.octron.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.octron.verticalLayoutWidget.setGeometry(QRect(0, 0, 412, 691))
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
        self.octron.toolBox.setMinimumSize(QSize(410, 550))
        self.octron.toolBox.setMaximumSize(QSize(410, 550))
        self.octron.toolBox.setCursor(QCursor(Qt.ArrowCursor))
        self.octron.toolBox.setFrameShape(QFrame.Shape.NoFrame)
        self.octron.toolBox.setFrameShadow(QFrame.Shadow.Plain)
        self.octron.toolBox.setLineWidth(0)
        self.octron.toolBox.setMidLineWidth(0)
        self.octron.project_tab = QWidget()
        self.octron.project_tab.setObjectName(u"project_tab")
        self.octron.project_tab.setGeometry(QRect(0, 0, 410, 414))
        sizePolicy1 = QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.octron.project_tab.sizePolicy().hasHeightForWidth())
        self.octron.project_tab.setSizePolicy(sizePolicy1)
        self.octron.verticalLayoutWidget_3 = QWidget(self.octron.project_tab)
        self.octron.verticalLayoutWidget_3.setObjectName(u"verticalLayoutWidget_3")
        self.octron.verticalLayoutWidget_3.setGeometry(QRect(0, -1, 402, 412))
        self.octron.project_vertical_layout = QVBoxLayout(self.octron.verticalLayoutWidget_3)
        self.octron.project_vertical_layout.setSpacing(20)
        self.octron.project_vertical_layout.setObjectName(u"project_vertical_layout")
        self.octron.project_vertical_layout.setContentsMargins(0, 0, 0, 15)
        self.octron.folder_sect_groupbox = QGroupBox(self.octron.verticalLayoutWidget_3)
        self.octron.folder_sect_groupbox.setObjectName(u"folder_sect_groupbox")
        sizePolicy2 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.octron.folder_sect_groupbox.sizePolicy().hasHeightForWidth())
        self.octron.folder_sect_groupbox.setSizePolicy(sizePolicy2)
        self.octron.folder_sect_groupbox.setMinimumSize(QSize(400, 80))
        self.octron.folder_sect_groupbox.setMaximumSize(QSize(400, 90))
        self.octron.horizontalLayout_11 = QHBoxLayout(self.octron.folder_sect_groupbox)
        self.octron.horizontalLayout_11.setSpacing(20)
        self.octron.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.octron.horizontalLayout_11.setContentsMargins(9, 12, 9, 12)
        self.octron.create_project_btn = QPushButton(self.octron.folder_sect_groupbox)
        self.octron.create_project_btn.setObjectName(u"create_project_btn")
        self.octron.create_project_btn.setMinimumSize(QSize(100, 25))
        self.octron.create_project_btn.setMaximumSize(QSize(100, 25))

        self.octron.horizontalLayout_11.addWidget(self.octron.create_project_btn, 0, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.octron.project_folder_path_label = QLabel(self.octron.folder_sect_groupbox)
        self.octron.project_folder_path_label.setObjectName(u"project_folder_path_label")
        self.octron.project_folder_path_label.setEnabled(False)
        self.octron.project_folder_path_label.setMinimumSize(QSize(250, 50))
        self.octron.project_folder_path_label.setMaximumSize(QSize(250, 50))
        self.octron.project_folder_path_label.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)
        self.octron.project_folder_path_label.setWordWrap(True)
        self.octron.project_folder_path_label.setMargin(0)

        self.octron.horizontalLayout_11.addWidget(self.octron.project_folder_path_label)


        self.octron.project_vertical_layout.addWidget(self.octron.folder_sect_groupbox, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop)

        self.octron.project_video_drop_groupbox = QGroupBox(self.octron.verticalLayoutWidget_3)
        self.octron.project_video_drop_groupbox.setObjectName(u"project_video_drop_groupbox")
        sizePolicy2.setHeightForWidth(self.octron.project_video_drop_groupbox.sizePolicy().hasHeightForWidth())
        self.octron.project_video_drop_groupbox.setSizePolicy(sizePolicy2)
        self.octron.project_video_drop_groupbox.setMinimumSize(QSize(400, 100))
        self.octron.project_video_drop_groupbox.setMaximumSize(QSize(400, 100))
        self.octron.horizontalLayout = QHBoxLayout(self.octron.project_video_drop_groupbox)
        self.octron.horizontalLayout.setSpacing(9)
        self.octron.horizontalLayout.setObjectName(u"horizontalLayout")
        self.octron.horizontalLayout.setContentsMargins(9, 12, 9, 12)
        self.octron.video_file_drop_widget = Mp4DropWidget()
        self.octron.video_file_drop_widget.setObjectName(u"video_file_drop_widget")
        self.octron.video_file_drop_widget.setMinimumSize(QSize(380, 60))
        self.octron.video_file_drop_widget.setMaximumSize(QSize(380, 60))

        self.octron.horizontalLayout.addWidget(self.octron.video_file_drop_widget)


        self.octron.project_vertical_layout.addWidget(self.octron.project_video_drop_groupbox)

        self.octron.project_existing_data_groupbox = QGroupBox(self.octron.verticalLayoutWidget_3)
        self.octron.project_existing_data_groupbox.setObjectName(u"project_existing_data_groupbox")
        self.octron.project_existing_data_groupbox.setEnabled(False)
        sizePolicy2.setHeightForWidth(self.octron.project_existing_data_groupbox.sizePolicy().hasHeightForWidth())
        self.octron.project_existing_data_groupbox.setSizePolicy(sizePolicy2)
        self.octron.project_existing_data_groupbox.setMinimumSize(QSize(400, 180))
        self.octron.project_existing_data_groupbox.setMaximumSize(QSize(400, 180))
        self.octron.horizontalLayout_9 = QHBoxLayout(self.octron.project_existing_data_groupbox)
        self.octron.horizontalLayout_9.setSpacing(20)
        self.octron.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.octron.horizontalLayout_9.setContentsMargins(9, 12, 9, 12)
        self.octron.existing_data_table = QTableView(self.octron.project_existing_data_groupbox)
        self.octron.existing_data_table.setObjectName(u"existing_data_table")
        self.octron.existing_data_table.setMinimumSize(QSize(380, 140))
        self.octron.existing_data_table.setMaximumSize(QSize(380, 130))
        self.octron.existing_data_table.setEditTriggers(QAbstractItemView.EditTrigger.AnyKeyPressed|QAbstractItemView.EditTrigger.EditKeyPressed|QAbstractItemView.EditTrigger.SelectedClicked)
        self.octron.existing_data_table.setProperty("showDropIndicator", False)
        self.octron.existing_data_table.setDragDropOverwriteMode(False)
        self.octron.existing_data_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.octron.existing_data_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.octron.existing_data_table.setGridStyle(Qt.PenStyle.SolidLine)
        self.octron.existing_data_table.setSortingEnabled(False)
        self.octron.existing_data_table.setWordWrap(False)
        self.octron.existing_data_table.setCornerButtonEnabled(False)
        self.octron.existing_data_table.horizontalHeader().setCascadingSectionResizes(True)
        self.octron.existing_data_table.horizontalHeader().setMinimumSectionSize(85)
        self.octron.existing_data_table.horizontalHeader().setDefaultSectionSize(85)
        self.octron.existing_data_table.horizontalHeader().setHighlightSections(False)
        self.octron.existing_data_table.verticalHeader().setVisible(False)
        self.octron.existing_data_table.verticalHeader().setMinimumSectionSize(20)
        self.octron.existing_data_table.verticalHeader().setDefaultSectionSize(20)
        self.octron.existing_data_table.verticalHeader().setHighlightSections(False)

        self.octron.horizontalLayout_9.addWidget(self.octron.existing_data_table, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignVCenter)


        self.octron.project_vertical_layout.addWidget(self.octron.project_existing_data_groupbox, 0, Qt.AlignmentFlag.AlignTop)

        icon = QIcon()
        icon.addFile(f"{base_path}/qt_gui/icons/noun-project-7158867.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.octron.toolBox.addItem(self.octron.project_tab, icon, u"Manage project")
        self.octron.annotate_tab = QWidget()
        self.octron.annotate_tab.setObjectName(u"annotate_tab")
        self.octron.annotate_tab.setGeometry(QRect(0, 0, 405, 414))
        sizePolicy1.setHeightForWidth(self.octron.annotate_tab.sizePolicy().hasHeightForWidth())
        self.octron.annotate_tab.setSizePolicy(sizePolicy1)
        self.octron.annotate_tab.setMaximumSize(QSize(405, 700))
        self.octron.verticalLayoutWidget_2 = QWidget(self.octron.annotate_tab)
        self.octron.verticalLayoutWidget_2.setObjectName(u"verticalLayoutWidget_2")
        self.octron.verticalLayoutWidget_2.setGeometry(QRect(0, 0, 402, 366))
        self.octron.annotate_vertical_layout = QVBoxLayout(self.octron.verticalLayoutWidget_2)
        self.octron.annotate_vertical_layout.setSpacing(20)
        self.octron.annotate_vertical_layout.setObjectName(u"annotate_vertical_layout")
        self.octron.annotate_vertical_layout.setContentsMargins(0, 0, 0, 15)
        self.octron.horizontalGroupBox = QGroupBox(self.octron.verticalLayoutWidget_2)
        self.octron.horizontalGroupBox.setObjectName(u"horizontalGroupBox")
        sizePolicy3 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.octron.horizontalGroupBox.sizePolicy().hasHeightForWidth())
        self.octron.horizontalGroupBox.setSizePolicy(sizePolicy3)
        self.octron.horizontalGroupBox.setMinimumSize(QSize(400, 70))
        self.octron.horizontalGroupBox.setMaximumSize(QSize(400, 70))
        self.octron.horizontalLayout_8 = QHBoxLayout(self.octron.horizontalGroupBox)
        self.octron.horizontalLayout_8.setSpacing(20)
        self.octron.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.octron.horizontalLayout_8.setContentsMargins(9, 12, 9, 12)
        self.octron.sam2model_list = QComboBox(self.octron.horizontalGroupBox)
        self.octron.sam2model_list.addItem("")
        self.octron.sam2model_list.setObjectName(u"sam2model_list")
        self.octron.sam2model_list.setMinimumSize(QSize(167, 25))
        self.octron.sam2model_list.setMaximumSize(QSize(167, 25))

        self.octron.horizontalLayout_8.addWidget(self.octron.sam2model_list, 0, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.octron.load_sam2model_btn = QPushButton(self.octron.horizontalGroupBox)
        self.octron.load_sam2model_btn.setObjectName(u"load_sam2model_btn")
        self.octron.load_sam2model_btn.setMinimumSize(QSize(0, 25))
        self.octron.load_sam2model_btn.setMaximumSize(QSize(250, 25))

        self.octron.horizontalLayout_8.addWidget(self.octron.load_sam2model_btn, 0, Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)


        self.octron.annotate_vertical_layout.addWidget(self.octron.horizontalGroupBox, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop)

        self.octron.annotate_layer_create_groupbox = QGroupBox(self.octron.verticalLayoutWidget_2)
        self.octron.annotate_layer_create_groupbox.setObjectName(u"annotate_layer_create_groupbox")
        sizePolicy3.setHeightForWidth(self.octron.annotate_layer_create_groupbox.sizePolicy().hasHeightForWidth())
        self.octron.annotate_layer_create_groupbox.setSizePolicy(sizePolicy3)
        self.octron.annotate_layer_create_groupbox.setMinimumSize(QSize(400, 100))
        self.octron.annotate_layer_create_groupbox.setMaximumSize(QSize(400, 100))
        self.octron.gridLayout = QGridLayout(self.octron.annotate_layer_create_groupbox)
        self.octron.gridLayout.setObjectName(u"gridLayout")
        self.octron.gridLayout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.octron.gridLayout.setHorizontalSpacing(20)
        self.octron.gridLayout.setVerticalSpacing(9)
        self.octron.gridLayout.setContentsMargins(9, 12, 9, 12)
        self.octron.create_projection_layer_btn = QPushButton(self.octron.annotate_layer_create_groupbox)
        self.octron.create_projection_layer_btn.setObjectName(u"create_projection_layer_btn")
        self.octron.create_projection_layer_btn.setMinimumSize(QSize(110, 25))
        self.octron.create_projection_layer_btn.setMaximumSize(QSize(110, 25))

        self.octron.gridLayout.addWidget(self.octron.create_projection_layer_btn, 1, 0, 1, 1, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

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

        self.octron.label_suffix_lineedit = QLineEdit(self.octron.annotate_layer_create_groupbox)
        self.octron.label_suffix_lineedit.setObjectName(u"label_suffix_lineedit")
        self.octron.label_suffix_lineedit.setMinimumSize(QSize(60, 25))
        self.octron.label_suffix_lineedit.setMaximumSize(QSize(60, 25))
        self.octron.label_suffix_lineedit.setInputMask(u"")
        self.octron.label_suffix_lineedit.setText(u"")
        self.octron.label_suffix_lineedit.setMaxLength(100)

        self.octron.gridLayout.addWidget(self.octron.label_suffix_lineedit, 0, 2, 1, 1, Qt.AlignmentFlag.AlignVCenter)

        self.octron.label_list_combobox = QComboBox(self.octron.annotate_layer_create_groupbox)
        self.octron.label_list_combobox.addItem("")
        self.octron.label_list_combobox.addItem("")
        self.octron.label_list_combobox.addItem("")
        self.octron.label_list_combobox.setObjectName(u"label_list_combobox")
        self.octron.label_list_combobox.setMinimumSize(QSize(110, 25))
        self.octron.label_list_combobox.setMaximumSize(QSize(110, 25))
        self.octron.label_list_combobox.setEditable(False)
        self.octron.label_list_combobox.setMaxVisibleItems(30)
        self.octron.label_list_combobox.setMaxCount(30)
        self.octron.label_list_combobox.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.octron.label_list_combobox.setIconSize(QSize(14, 14))
        self.octron.label_list_combobox.setFrame(False)

        self.octron.gridLayout.addWidget(self.octron.label_list_combobox, 0, 1, 1, 1, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.octron.hard_reset_layer_btn = QPushButton(self.octron.annotate_layer_create_groupbox)
        self.octron.hard_reset_layer_btn.setObjectName(u"hard_reset_layer_btn")
        self.octron.hard_reset_layer_btn.setMinimumSize(QSize(70, 25))
        self.octron.hard_reset_layer_btn.setMaximumSize(QSize(70, 25))
        self.octron.hard_reset_layer_btn.setAutoRepeatInterval(2000)

        self.octron.gridLayout.addWidget(self.octron.hard_reset_layer_btn, 1, 3, 1, 1, Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)

        self.octron.create_annotation_layer_btn = QPushButton(self.octron.annotate_layer_create_groupbox)
        self.octron.create_annotation_layer_btn.setObjectName(u"create_annotation_layer_btn")
        self.octron.create_annotation_layer_btn.setMinimumSize(QSize(70, 25))
        self.octron.create_annotation_layer_btn.setMaximumSize(QSize(70, 25))

        self.octron.gridLayout.addWidget(self.octron.create_annotation_layer_btn, 0, 3, 1, 1, Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)


        self.octron.annotate_vertical_layout.addWidget(self.octron.annotate_layer_create_groupbox, 0, Qt.AlignmentFlag.AlignTop)

        self.octron.annotate_layer_timeline_groupbox = QGroupBox(self.octron.verticalLayoutWidget_2)
        self.octron.annotate_layer_timeline_groupbox.setObjectName(u"annotate_layer_timeline_groupbox")
        sizePolicy3.setHeightForWidth(self.octron.annotate_layer_timeline_groupbox.sizePolicy().hasHeightForWidth())
        self.octron.annotate_layer_timeline_groupbox.setSizePolicy(sizePolicy3)
        self.octron.annotate_layer_timeline_groupbox.setMinimumSize(QSize(400, 60))
        self.octron.annotate_layer_timeline_groupbox.setMaximumSize(QSize(400, 60))
        self.octron.gridLayout_4 = QGridLayout(self.octron.annotate_layer_timeline_groupbox)
        self.octron.gridLayout_4.setObjectName(u"gridLayout_4")
        self.octron.gridLayout_4.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.octron.gridLayout_4.setHorizontalSpacing(20)
        self.octron.gridLayout_4.setVerticalSpacing(9)
        self.octron.gridLayout_4.setContentsMargins(9, 12, 9, 12)
        self.octron.annotation_jump_previous_btn = QPushButton(self.octron.annotate_layer_timeline_groupbox)
        self.octron.annotation_jump_previous_btn.setObjectName(u"annotation_jump_previous_btn")
        self.octron.annotation_jump_previous_btn.setMinimumSize(QSize(150, 25))
        self.octron.annotation_jump_previous_btn.setMaximumSize(QSize(70, 25))

        self.octron.gridLayout_4.addWidget(self.octron.annotation_jump_previous_btn, 0, 0, 1, 1, Qt.AlignmentFlag.AlignVCenter)

        self.octron.annotation_jump_next_btn = QPushButton(self.octron.annotate_layer_timeline_groupbox)
        self.octron.annotation_jump_next_btn.setObjectName(u"annotation_jump_next_btn")
        self.octron.annotation_jump_next_btn.setMinimumSize(QSize(150, 25))
        self.octron.annotation_jump_next_btn.setMaximumSize(QSize(70, 25))

        self.octron.gridLayout_4.addWidget(self.octron.annotation_jump_next_btn, 0, 1, 1, 1, Qt.AlignmentFlag.AlignVCenter)


        self.octron.annotate_vertical_layout.addWidget(self.octron.annotate_layer_timeline_groupbox)

        self.octron.annotate_layer_predict_groupbox = QGroupBox(self.octron.verticalLayoutWidget_2)
        self.octron.annotate_layer_predict_groupbox.setObjectName(u"annotate_layer_predict_groupbox")
        sizePolicy3.setHeightForWidth(self.octron.annotate_layer_predict_groupbox.sizePolicy().hasHeightForWidth())
        self.octron.annotate_layer_predict_groupbox.setSizePolicy(sizePolicy3)
        self.octron.annotate_layer_predict_groupbox.setMinimumSize(QSize(400, 70))
        self.octron.annotate_layer_predict_groupbox.setMaximumSize(QSize(400, 70))
        self.octron.horizontalLayout_2 = QHBoxLayout(self.octron.annotate_layer_predict_groupbox)
        self.octron.horizontalLayout_2.setSpacing(5)
        self.octron.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.octron.horizontalLayout_2.setContentsMargins(9, 12, 9, 12)
        self.octron.batch_predict_progressbar = QProgressBar(self.octron.annotate_layer_predict_groupbox)
        self.octron.batch_predict_progressbar.setObjectName(u"batch_predict_progressbar")
        sizePolicy4 = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.octron.batch_predict_progressbar.sizePolicy().hasHeightForWidth())
        self.octron.batch_predict_progressbar.setSizePolicy(sizePolicy4)
        self.octron.batch_predict_progressbar.setMinimumSize(QSize(130, 25))
        self.octron.batch_predict_progressbar.setMaximumSize(QSize(130, 25))
        self.octron.batch_predict_progressbar.setMaximum(20)
        self.octron.batch_predict_progressbar.setValue(0)

        self.octron.horizontalLayout_2.addWidget(self.octron.batch_predict_progressbar, 0, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.octron.skip_label = QLabel(self.octron.annotate_layer_predict_groupbox)
        self.octron.skip_label.setObjectName(u"skip_label")
        self.octron.skip_label.setMinimumSize(QSize(30, 25))
        self.octron.skip_label.setMaximumSize(QSize(30, 25))
        self.octron.skip_label.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.octron.horizontalLayout_2.addWidget(self.octron.skip_label, 0, Qt.AlignmentFlag.AlignRight)

        self.octron.skip_frames_spinbox = QSpinBox(self.octron.annotate_layer_predict_groupbox)
        self.octron.skip_frames_spinbox.setObjectName(u"skip_frames_spinbox")
        sizePolicy3.setHeightForWidth(self.octron.skip_frames_spinbox.sizePolicy().hasHeightForWidth())
        self.octron.skip_frames_spinbox.setSizePolicy(sizePolicy3)
        self.octron.skip_frames_spinbox.setMinimumSize(QSize(35, 25))
        self.octron.skip_frames_spinbox.setMaximumSize(QSize(35, 25))
        self.octron.skip_frames_spinbox.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)
        self.octron.skip_frames_spinbox.setMaximum(200)

        self.octron.horizontalLayout_2.addWidget(self.octron.skip_frames_spinbox)

        self.octron.predict_next_oneframe_btn = QPushButton(self.octron.annotate_layer_predict_groupbox)
        self.octron.predict_next_oneframe_btn.setObjectName(u"predict_next_oneframe_btn")
        self.octron.predict_next_oneframe_btn.setEnabled(False)
        sizePolicy3.setHeightForWidth(self.octron.predict_next_oneframe_btn.sizePolicy().hasHeightForWidth())
        self.octron.predict_next_oneframe_btn.setSizePolicy(sizePolicy3)
        self.octron.predict_next_oneframe_btn.setMinimumSize(QSize(20, 25))
        self.octron.predict_next_oneframe_btn.setMaximumSize(QSize(20, 25))
        self.octron.predict_next_oneframe_btn.setBaseSize(QSize(15, 25))

        self.octron.horizontalLayout_2.addWidget(self.octron.predict_next_oneframe_btn)

        self.octron.predict_next_batch_btn = QPushButton(self.octron.annotate_layer_predict_groupbox)
        self.octron.predict_next_batch_btn.setObjectName(u"predict_next_batch_btn")
        self.octron.predict_next_batch_btn.setEnabled(False)
        self.octron.predict_next_batch_btn.setMinimumSize(QSize(80, 25))
        self.octron.predict_next_batch_btn.setMaximumSize(QSize(80, 25))

        self.octron.horizontalLayout_2.addWidget(self.octron.predict_next_batch_btn, 0, Qt.AlignmentFlag.AlignVCenter)


        self.octron.annotate_vertical_layout.addWidget(self.octron.annotate_layer_predict_groupbox, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignBottom)

        icon1 = QIcon()
        icon1.addFile(f"{base_path}/qt_gui/icons/noun-copywriting-7158879.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.octron.toolBox.addItem(self.octron.annotate_tab, icon1, u"Generate annotation data")
        self.octron.train_tab = QWidget()
        self.octron.train_tab.setObjectName(u"train_tab")
        self.octron.train_tab.setGeometry(QRect(0, 0, 410, 414))
        sizePolicy1.setHeightForWidth(self.octron.train_tab.sizePolicy().hasHeightForWidth())
        self.octron.train_tab.setSizePolicy(sizePolicy1)
        self.octron.verticalLayoutWidget_4 = QWidget(self.octron.train_tab)
        self.octron.verticalLayoutWidget_4.setObjectName(u"verticalLayoutWidget_4")
        self.octron.verticalLayoutWidget_4.setGeometry(QRect(0, 0, 402, 371))
        self.octron.train_vertical_layout = QVBoxLayout(self.octron.verticalLayoutWidget_4)
        self.octron.train_vertical_layout.setObjectName(u"train_vertical_layout")
        self.octron.train_vertical_layout.setContentsMargins(0, 0, 0, 0)
        self.octron.train_generate_groupbox = QGroupBox(self.octron.verticalLayoutWidget_4)
        self.octron.train_generate_groupbox.setObjectName(u"train_generate_groupbox")
        self.octron.train_generate_groupbox.setEnabled(False)
        sizePolicy3.setHeightForWidth(self.octron.train_generate_groupbox.sizePolicy().hasHeightForWidth())
        self.octron.train_generate_groupbox.setSizePolicy(sizePolicy3)
        self.octron.train_generate_groupbox.setMinimumSize(QSize(400, 160))
        self.octron.train_generate_groupbox.setMaximumSize(QSize(400, 140))
        self.octron.train_generate_groupbox.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
        self.octron.layoutWidget = QWidget(self.octron.train_generate_groupbox)
        self.octron.layoutWidget.setObjectName(u"layoutWidget")
        self.octron.layoutWidget.setGeometry(QRect(10, 30, 281, 29))
        self.octron.train_progress_A_horizontalLayout = QHBoxLayout(self.octron.layoutWidget)
        self.octron.train_progress_A_horizontalLayout.setObjectName(u"train_progress_A_horizontalLayout")
        self.octron.train_progress_A_horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.octron.train_polygons_overall_progressbar = QProgressBar(self.octron.layoutWidget)
        self.octron.train_polygons_overall_progressbar.setObjectName(u"train_polygons_overall_progressbar")
        self.octron.train_polygons_overall_progressbar.setEnabled(False)
        sizePolicy4.setHeightForWidth(self.octron.train_polygons_overall_progressbar.sizePolicy().hasHeightForWidth())
        self.octron.train_polygons_overall_progressbar.setSizePolicy(sizePolicy4)
        self.octron.train_polygons_overall_progressbar.setMinimumSize(QSize(50, 25))
        self.octron.train_polygons_overall_progressbar.setMaximumSize(QSize(50, 25))
        self.octron.train_polygons_overall_progressbar.setMaximum(20)
        self.octron.train_polygons_overall_progressbar.setValue(0)

        self.octron.train_progress_A_horizontalLayout.addWidget(self.octron.train_polygons_overall_progressbar)

        self.octron.train_polygons_frames_progressbar = QProgressBar(self.octron.layoutWidget)
        self.octron.train_polygons_frames_progressbar.setObjectName(u"train_polygons_frames_progressbar")
        self.octron.train_polygons_frames_progressbar.setEnabled(False)
        sizePolicy4.setHeightForWidth(self.octron.train_polygons_frames_progressbar.sizePolicy().hasHeightForWidth())
        self.octron.train_polygons_frames_progressbar.setSizePolicy(sizePolicy4)
        self.octron.train_polygons_frames_progressbar.setMinimumSize(QSize(100, 25))
        self.octron.train_polygons_frames_progressbar.setMaximumSize(QSize(100, 25))
        self.octron.train_polygons_frames_progressbar.setMaximum(20)
        self.octron.train_polygons_frames_progressbar.setValue(0)

        self.octron.train_progress_A_horizontalLayout.addWidget(self.octron.train_polygons_frames_progressbar)

        self.octron.train_polygons_label = QLabel(self.octron.layoutWidget)
        self.octron.train_polygons_label.setObjectName(u"train_polygons_label")
        self.octron.train_polygons_label.setEnabled(False)
        self.octron.train_polygons_label.setMinimumSize(QSize(0, 25))
        self.octron.train_polygons_label.setMaximumSize(QSize(16777215, 25))

        self.octron.train_progress_A_horizontalLayout.addWidget(self.octron.train_polygons_label)

        self.octron.layoutWidget1 = QWidget(self.octron.train_generate_groupbox)
        self.octron.layoutWidget1.setObjectName(u"layoutWidget1")
        self.octron.layoutWidget1.setGeometry(QRect(10, 60, 281, 29))
        self.octron.train_progress_B_horizontalLayout = QHBoxLayout(self.octron.layoutWidget1)
        self.octron.train_progress_B_horizontalLayout.setObjectName(u"train_progress_B_horizontalLayout")
        self.octron.train_progress_B_horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.octron.train_export_overall_progressbar = QProgressBar(self.octron.layoutWidget1)
        self.octron.train_export_overall_progressbar.setObjectName(u"train_export_overall_progressbar")
        self.octron.train_export_overall_progressbar.setEnabled(False)
        sizePolicy4.setHeightForWidth(self.octron.train_export_overall_progressbar.sizePolicy().hasHeightForWidth())
        self.octron.train_export_overall_progressbar.setSizePolicy(sizePolicy4)
        self.octron.train_export_overall_progressbar.setMinimumSize(QSize(50, 25))
        self.octron.train_export_overall_progressbar.setMaximumSize(QSize(50, 25))
        self.octron.train_export_overall_progressbar.setMaximum(20)
        self.octron.train_export_overall_progressbar.setValue(0)

        self.octron.train_progress_B_horizontalLayout.addWidget(self.octron.train_export_overall_progressbar)

        self.octron.train_export_frames_progressbar = QProgressBar(self.octron.layoutWidget1)
        self.octron.train_export_frames_progressbar.setObjectName(u"train_export_frames_progressbar")
        self.octron.train_export_frames_progressbar.setEnabled(False)
        sizePolicy4.setHeightForWidth(self.octron.train_export_frames_progressbar.sizePolicy().hasHeightForWidth())
        self.octron.train_export_frames_progressbar.setSizePolicy(sizePolicy4)
        self.octron.train_export_frames_progressbar.setMinimumSize(QSize(100, 25))
        self.octron.train_export_frames_progressbar.setMaximumSize(QSize(100, 25))
        self.octron.train_export_frames_progressbar.setMaximum(20)
        self.octron.train_export_frames_progressbar.setValue(0)

        self.octron.train_progress_B_horizontalLayout.addWidget(self.octron.train_export_frames_progressbar)

        self.octron.train_export_label = QLabel(self.octron.layoutWidget1)
        self.octron.train_export_label.setObjectName(u"train_export_label")
        self.octron.train_export_label.setEnabled(False)
        self.octron.train_export_label.setMinimumSize(QSize(0, 25))
        self.octron.train_export_label.setMaximumSize(QSize(16777215, 25))

        self.octron.train_progress_B_horizontalLayout.addWidget(self.octron.train_export_label)

        self.octron.layoutWidget2 = QWidget(self.octron.train_generate_groupbox)
        self.octron.layoutWidget2.setObjectName(u"layoutWidget2")
        self.octron.layoutWidget2.setGeometry(QRect(300, 30, 90, 81))
        self.octron.train_checkboxes_verticalLayout = QVBoxLayout(self.octron.layoutWidget2)
        self.octron.train_checkboxes_verticalLayout.setSpacing(10)
        self.octron.train_checkboxes_verticalLayout.setObjectName(u"train_checkboxes_verticalLayout")
        self.octron.train_checkboxes_verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.octron.train_prune_checkBox = QCheckBox(self.octron.layoutWidget2)
        self.octron.train_prune_checkBox.setObjectName(u"train_prune_checkBox")
        self.octron.train_prune_checkBox.setEnabled(False)
        self.octron.train_prune_checkBox.setMinimumSize(QSize(90, 25))
        self.octron.train_prune_checkBox.setMaximumSize(QSize(90, 25))
        self.octron.train_prune_checkBox.setChecked(False)

        self.octron.train_checkboxes_verticalLayout.addWidget(self.octron.train_prune_checkBox)

        self.octron.train_data_watershed_checkBox = QCheckBox(self.octron.layoutWidget2)
        self.octron.train_data_watershed_checkBox.setObjectName(u"train_data_watershed_checkBox")
        self.octron.train_data_watershed_checkBox.setEnabled(False)
        self.octron.train_data_watershed_checkBox.setMinimumSize(QSize(90, 25))
        self.octron.train_data_watershed_checkBox.setMaximumSize(QSize(90, 25))
        self.octron.train_data_watershed_checkBox.setChecked(False)

        self.octron.train_checkboxes_verticalLayout.addWidget(self.octron.train_data_watershed_checkBox)

        self.octron.train_data_overwrite_checkBox = QCheckBox(self.octron.layoutWidget2)
        self.octron.train_data_overwrite_checkBox.setObjectName(u"train_data_overwrite_checkBox")
        self.octron.train_data_overwrite_checkBox.setEnabled(False)
        self.octron.train_data_overwrite_checkBox.setMinimumSize(QSize(90, 25))
        self.octron.train_data_overwrite_checkBox.setMaximumSize(QSize(90, 25))
        self.octron.train_data_overwrite_checkBox.setChecked(True)

        self.octron.train_checkboxes_verticalLayout.addWidget(self.octron.train_data_overwrite_checkBox)

        self.octron.layoutWidget3 = QWidget(self.octron.train_generate_groupbox)
        self.octron.layoutWidget3.setObjectName(u"layoutWidget3")
        self.octron.layoutWidget3.setGeometry(QRect(10, 120, 381, 37))
        self.octron.train_folder_btn_horizontalLayout = QHBoxLayout(self.octron.layoutWidget3)
        self.octron.train_folder_btn_horizontalLayout.setObjectName(u"train_folder_btn_horizontalLayout")
        self.octron.train_folder_btn_horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.octron.training_data_folder_label = QLabel(self.octron.layoutWidget3)
        self.octron.training_data_folder_label.setObjectName(u"training_data_folder_label")
        self.octron.training_data_folder_label.setEnabled(False)
        self.octron.training_data_folder_label.setMinimumSize(QSize(275, 25))
        self.octron.training_data_folder_label.setMaximumSize(QSize(275, 25))

        self.octron.train_folder_btn_horizontalLayout.addWidget(self.octron.training_data_folder_label)

        self.octron.generate_training_data_btn = QPushButton(self.octron.layoutWidget3)
        self.octron.generate_training_data_btn.setObjectName(u"generate_training_data_btn")
        sizePolicy3.setHeightForWidth(self.octron.generate_training_data_btn.sizePolicy().hasHeightForWidth())
        self.octron.generate_training_data_btn.setSizePolicy(sizePolicy3)
        self.octron.generate_training_data_btn.setMinimumSize(QSize(90, 25))
        self.octron.generate_training_data_btn.setMaximumSize(QSize(90, 25))

        self.octron.train_folder_btn_horizontalLayout.addWidget(self.octron.generate_training_data_btn, 0, Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)


        self.octron.train_vertical_layout.addWidget(self.octron.train_generate_groupbox, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop)

        self.octron.train_train_groupbox = QGroupBox(self.octron.verticalLayoutWidget_4)
        self.octron.train_train_groupbox.setObjectName(u"train_train_groupbox")
        self.octron.train_train_groupbox.setEnabled(False)
        sizePolicy3.setHeightForWidth(self.octron.train_train_groupbox.sizePolicy().hasHeightForWidth())
        self.octron.train_train_groupbox.setSizePolicy(sizePolicy3)
        self.octron.train_train_groupbox.setMinimumSize(QSize(400, 185))
        self.octron.train_train_groupbox.setMaximumSize(QSize(400, 185))
        self.octron.train_train_groupbox.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
        self.octron.layoutWidget4 = QWidget(self.octron.train_train_groupbox)
        self.octron.layoutWidget4.setObjectName(u"layoutWidget4")
        self.octron.layoutWidget4.setGeometry(QRect(10, 70, 201, 62))
        self.octron.train_grid_layout = QGridLayout(self.octron.layoutWidget4)
        self.octron.train_grid_layout.setObjectName(u"train_grid_layout")
        self.octron.train_grid_layout.setContentsMargins(0, 0, 10, 0)
        self.octron.num_epochs_label = QLabel(self.octron.layoutWidget4)
        self.octron.num_epochs_label.setObjectName(u"num_epochs_label")
        sizePolicy5 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.octron.num_epochs_label.sizePolicy().hasHeightForWidth())
        self.octron.num_epochs_label.setSizePolicy(sizePolicy5)
        self.octron.num_epochs_label.setMinimumSize(QSize(100, 0))
        self.octron.num_epochs_label.setMaximumSize(QSize(100, 25))

        self.octron.train_grid_layout.addWidget(self.octron.num_epochs_label, 0, 0, 1, 1, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.octron.save_period_label = QLabel(self.octron.layoutWidget4)
        self.octron.save_period_label.setObjectName(u"save_period_label")
        sizePolicy5.setHeightForWidth(self.octron.save_period_label.sizePolicy().hasHeightForWidth())
        self.octron.save_period_label.setSizePolicy(sizePolicy5)
        self.octron.save_period_label.setMinimumSize(QSize(100, 0))
        self.octron.save_period_label.setMaximumSize(QSize(100, 25))

        self.octron.train_grid_layout.addWidget(self.octron.save_period_label, 1, 0, 1, 1, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.octron.num_epochs_input = QSpinBox(self.octron.layoutWidget4)
        self.octron.num_epochs_input.setObjectName(u"num_epochs_input")
        sizePolicy3.setHeightForWidth(self.octron.num_epochs_input.sizePolicy().hasHeightForWidth())
        self.octron.num_epochs_input.setSizePolicy(sizePolicy3)
        self.octron.num_epochs_input.setMinimumSize(QSize(80, 25))
        self.octron.num_epochs_input.setMaximumSize(QSize(80, 25))
        self.octron.num_epochs_input.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)
        self.octron.num_epochs_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.PlusMinus)
        self.octron.num_epochs_input.setMinimum(1)
        self.octron.num_epochs_input.setMaximum(900)
        self.octron.num_epochs_input.setSingleStep(10)
        self.octron.num_epochs_input.setValue(250)

        self.octron.train_grid_layout.addWidget(self.octron.num_epochs_input, 0, 1, 1, 1, Qt.AlignmentFlag.AlignRight)

        self.octron.save_period_input = QSpinBox(self.octron.layoutWidget4)
        self.octron.save_period_input.setObjectName(u"save_period_input")
        sizePolicy3.setHeightForWidth(self.octron.save_period_input.sizePolicy().hasHeightForWidth())
        self.octron.save_period_input.setSizePolicy(sizePolicy3)
        self.octron.save_period_input.setMinimumSize(QSize(80, 25))
        self.octron.save_period_input.setMaximumSize(QSize(80, 25))
        self.octron.save_period_input.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)
        self.octron.save_period_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.PlusMinus)
        self.octron.save_period_input.setMinimum(2)
        self.octron.save_period_input.setMaximum(100)
        self.octron.save_period_input.setSingleStep(15)
        self.octron.save_period_input.setValue(50)

        self.octron.train_grid_layout.addWidget(self.octron.save_period_input, 1, 1, 1, 1, Qt.AlignmentFlag.AlignRight)

        self.octron.layoutWidget5 = QWidget(self.octron.train_train_groupbox)
        self.octron.layoutWidget5.setObjectName(u"layoutWidget5")
        self.octron.layoutWidget5.setGeometry(QRect(300, 30, 90, 81))
        self.octron.train_verticalLayout = QVBoxLayout(self.octron.layoutWidget5)
        self.octron.train_verticalLayout.setSpacing(10)
        self.octron.train_verticalLayout.setObjectName(u"train_verticalLayout")
        self.octron.train_verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.octron.train_resume_checkBox = QCheckBox(self.octron.layoutWidget5)
        self.octron.train_resume_checkBox.setObjectName(u"train_resume_checkBox")
        self.octron.train_resume_checkBox.setEnabled(False)
        self.octron.train_resume_checkBox.setMinimumSize(QSize(90, 25))
        self.octron.train_resume_checkBox.setMaximumSize(QSize(90, 25))

        self.octron.train_verticalLayout.addWidget(self.octron.train_resume_checkBox, 0, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop)

        self.octron.train_training_overwrite_checkBox = QCheckBox(self.octron.layoutWidget5)
        self.octron.train_training_overwrite_checkBox.setObjectName(u"train_training_overwrite_checkBox")
        self.octron.train_training_overwrite_checkBox.setEnabled(False)
        self.octron.train_training_overwrite_checkBox.setMinimumSize(QSize(90, 25))
        self.octron.train_training_overwrite_checkBox.setMaximumSize(QSize(90, 25))
        self.octron.train_training_overwrite_checkBox.setChecked(True)

        self.octron.train_verticalLayout.addWidget(self.octron.train_training_overwrite_checkBox, 0, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop)

        self.octron.launch_tensorboard_checkBox = QCheckBox(self.octron.layoutWidget5)
        self.octron.launch_tensorboard_checkBox.setObjectName(u"launch_tensorboard_checkBox")
        self.octron.launch_tensorboard_checkBox.setEnabled(False)
        self.octron.launch_tensorboard_checkBox.setMinimumSize(QSize(90, 25))
        self.octron.launch_tensorboard_checkBox.setMaximumSize(QSize(90, 25))
        self.octron.launch_tensorboard_checkBox.setChecked(True)

        self.octron.train_verticalLayout.addWidget(self.octron.launch_tensorboard_checkBox)

        self.octron.layoutWidget6 = QWidget(self.octron.train_train_groupbox)
        self.octron.layoutWidget6.setObjectName(u"layoutWidget6")
        self.octron.layoutWidget6.setGeometry(QRect(10, 30, 281, 31))
        self.octron.model_choose_horizontalLayout = QHBoxLayout(self.octron.layoutWidget6)
        self.octron.model_choose_horizontalLayout.setObjectName(u"model_choose_horizontalLayout")
        self.octron.model_choose_horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.octron.yolomodel_list = QComboBox(self.octron.layoutWidget6)
        self.octron.yolomodel_list.addItem("")
        self.octron.yolomodel_list.setObjectName(u"yolomodel_list")
        self.octron.yolomodel_list.setMinimumSize(QSize(150, 25))
        self.octron.yolomodel_list.setMaximumSize(QSize(150, 25))

        self.octron.model_choose_horizontalLayout.addWidget(self.octron.yolomodel_list, 0, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.octron.yoloimagesize_list = QComboBox(self.octron.layoutWidget6)
        self.octron.yoloimagesize_list.addItem("")
        self.octron.yoloimagesize_list.addItem("")
        self.octron.yoloimagesize_list.addItem("")
        self.octron.yoloimagesize_list.setObjectName(u"yoloimagesize_list")
        self.octron.yoloimagesize_list.setMinimumSize(QSize(100, 25))
        self.octron.yoloimagesize_list.setMaximumSize(QSize(100, 25))
        self.octron.yoloimagesize_list.setEditable(True)

        self.octron.model_choose_horizontalLayout.addWidget(self.octron.yoloimagesize_list, 0, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.octron.layoutWidget7 = QWidget(self.octron.train_train_groupbox)
        self.octron.layoutWidget7.setObjectName(u"layoutWidget7")
        self.octron.layoutWidget7.setGeometry(QRect(10, 140, 380, 37))
        self.octron.epochs_horizontalLayout = QHBoxLayout(self.octron.layoutWidget7)
        self.octron.epochs_horizontalLayout.setObjectName(u"epochs_horizontalLayout")
        self.octron.epochs_horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.octron.train_epochs_progressbar = QProgressBar(self.octron.layoutWidget7)
        self.octron.train_epochs_progressbar.setObjectName(u"train_epochs_progressbar")
        self.octron.train_epochs_progressbar.setEnabled(False)
        sizePolicy4.setHeightForWidth(self.octron.train_epochs_progressbar.sizePolicy().hasHeightForWidth())
        self.octron.train_epochs_progressbar.setSizePolicy(sizePolicy4)
        self.octron.train_epochs_progressbar.setMinimumSize(QSize(120, 25))
        self.octron.train_epochs_progressbar.setMaximumSize(QSize(120, 25))
        self.octron.train_epochs_progressbar.setMaximum(20)
        self.octron.train_epochs_progressbar.setValue(0)

        self.octron.epochs_horizontalLayout.addWidget(self.octron.train_epochs_progressbar, 0, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.octron.train_finishtime_label = QLabel(self.octron.layoutWidget7)
        self.octron.train_finishtime_label.setObjectName(u"train_finishtime_label")
        self.octron.train_finishtime_label.setEnabled(False)
        self.octron.train_finishtime_label.setMinimumSize(QSize(150, 25))
        self.octron.train_finishtime_label.setMaximumSize(QSize(150, 25))

        self.octron.epochs_horizontalLayout.addWidget(self.octron.train_finishtime_label, 0, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.octron.start_stop_training_btn = QPushButton(self.octron.layoutWidget7)
        self.octron.start_stop_training_btn.setObjectName(u"start_stop_training_btn")
        sizePolicy3.setHeightForWidth(self.octron.start_stop_training_btn.sizePolicy().hasHeightForWidth())
        self.octron.start_stop_training_btn.setSizePolicy(sizePolicy3)
        self.octron.start_stop_training_btn.setMinimumSize(QSize(90, 25))
        self.octron.start_stop_training_btn.setMaximumSize(QSize(90, 25))

        self.octron.epochs_horizontalLayout.addWidget(self.octron.start_stop_training_btn, 0, Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)


        self.octron.train_vertical_layout.addWidget(self.octron.train_train_groupbox)

        icon2 = QIcon()
        icon2.addFile(f"{base_path}/qt_gui/icons/noun-rocket-7158872.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.octron.toolBox.addItem(self.octron.train_tab, icon2, u"Train model")
        self.octron.predict_tab = QWidget()
        self.octron.predict_tab.setObjectName(u"predict_tab")
        self.octron.predict_tab.setGeometry(QRect(0, 0, 410, 414))
        sizePolicy1.setHeightForWidth(self.octron.predict_tab.sizePolicy().hasHeightForWidth())
        self.octron.predict_tab.setSizePolicy(sizePolicy1)
        self.octron.verticalLayoutWidget_5 = QWidget(self.octron.predict_tab)
        self.octron.verticalLayoutWidget_5.setObjectName(u"verticalLayoutWidget_5")
        self.octron.verticalLayoutWidget_5.setGeometry(QRect(0, 0, 402, 409))
        self.octron.predict_verticalLayout = QVBoxLayout(self.octron.verticalLayoutWidget_5)
        self.octron.predict_verticalLayout.setSpacing(20)
        self.octron.predict_verticalLayout.setObjectName(u"predict_verticalLayout")
        self.octron.predict_verticalLayout.setContentsMargins(0, 0, 0, 10)
        self.octron.predict_video_drop_groupbox = QGroupBox(self.octron.verticalLayoutWidget_5)
        self.octron.predict_video_drop_groupbox.setObjectName(u"predict_video_drop_groupbox")
        self.octron.predict_video_drop_groupbox.setEnabled(True)
        sizePolicy2.setHeightForWidth(self.octron.predict_video_drop_groupbox.sizePolicy().hasHeightForWidth())
        self.octron.predict_video_drop_groupbox.setSizePolicy(sizePolicy2)
        self.octron.predict_video_drop_groupbox.setMinimumSize(QSize(400, 100))
        self.octron.predict_video_drop_groupbox.setMaximumSize(QSize(400, 100))
        self.octron.horizontalLayout_3 = QHBoxLayout(self.octron.predict_video_drop_groupbox)
        self.octron.horizontalLayout_3.setSpacing(20)
        self.octron.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.octron.horizontalLayout_3.setContentsMargins(9, 9, 9, 9)
        self.octron.predict_video_drop_widget = Mp4DropWidget()
        self.octron.predict_video_drop_widget.setObjectName(u"predict_video_drop_widget")
        self.octron.predict_video_drop_widget.setMinimumSize(QSize(380, 60))
        self.octron.predict_video_drop_widget.setMaximumSize(QSize(380, 60))

        self.octron.horizontalLayout_3.addWidget(self.octron.predict_video_drop_widget)


        self.octron.predict_verticalLayout.addWidget(self.octron.predict_video_drop_groupbox)

        self.octron.predict_video_predict_groupbox = QGroupBox(self.octron.verticalLayoutWidget_5)
        self.octron.predict_video_predict_groupbox.setObjectName(u"predict_video_predict_groupbox")
        self.octron.predict_video_predict_groupbox.setEnabled(True)
        sizePolicy2.setHeightForWidth(self.octron.predict_video_predict_groupbox.sizePolicy().hasHeightForWidth())
        self.octron.predict_video_predict_groupbox.setSizePolicy(sizePolicy2)
        self.octron.predict_video_predict_groupbox.setMinimumSize(QSize(400, 280))
        self.octron.predict_video_predict_groupbox.setMaximumSize(QSize(400, 280))
        self.octron.layoutWidget_2 = QWidget(self.octron.predict_video_predict_groupbox)
        self.octron.layoutWidget_2.setObjectName(u"layoutWidget_2")
        self.octron.layoutWidget_2.setGeometry(QRect(11, 193, 382, 29))
        self.octron.predict_progress_bar_layout = QHBoxLayout(self.octron.layoutWidget_2)
        self.octron.predict_progress_bar_layout.setObjectName(u"predict_progress_bar_layout")
        self.octron.predict_progress_bar_layout.setContentsMargins(0, 0, 0, 0)
        self.octron.predict_overall_progressbar = QProgressBar(self.octron.layoutWidget_2)
        self.octron.predict_overall_progressbar.setObjectName(u"predict_overall_progressbar")
        self.octron.predict_overall_progressbar.setEnabled(False)
        sizePolicy4.setHeightForWidth(self.octron.predict_overall_progressbar.sizePolicy().hasHeightForWidth())
        self.octron.predict_overall_progressbar.setSizePolicy(sizePolicy4)
        self.octron.predict_overall_progressbar.setMinimumSize(QSize(50, 25))
        self.octron.predict_overall_progressbar.setMaximumSize(QSize(50, 25))
        self.octron.predict_overall_progressbar.setMaximum(20)
        self.octron.predict_overall_progressbar.setValue(0)

        self.octron.predict_progress_bar_layout.addWidget(self.octron.predict_overall_progressbar, 0, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.octron.predict_current_video_progressbar = QProgressBar(self.octron.layoutWidget_2)
        self.octron.predict_current_video_progressbar.setObjectName(u"predict_current_video_progressbar")
        self.octron.predict_current_video_progressbar.setEnabled(False)
        sizePolicy4.setHeightForWidth(self.octron.predict_current_video_progressbar.sizePolicy().hasHeightForWidth())
        self.octron.predict_current_video_progressbar.setSizePolicy(sizePolicy4)
        self.octron.predict_current_video_progressbar.setMinimumSize(QSize(120, 25))
        self.octron.predict_current_video_progressbar.setMaximumSize(QSize(120, 25))
        self.octron.predict_current_video_progressbar.setMaximum(20)
        self.octron.predict_current_video_progressbar.setValue(0)

        self.octron.predict_progress_bar_layout.addWidget(self.octron.predict_current_video_progressbar, 0, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.octron.predict_current_videoname_label = QLabel(self.octron.layoutWidget_2)
        self.octron.predict_current_videoname_label.setObjectName(u"predict_current_videoname_label")
        self.octron.predict_current_videoname_label.setEnabled(False)
        self.octron.predict_current_videoname_label.setMinimumSize(QSize(188, 25))
        self.octron.predict_current_videoname_label.setMaximumSize(QSize(188, 25))

        self.octron.predict_progress_bar_layout.addWidget(self.octron.predict_current_videoname_label)

        self.octron.layoutWidget_3 = QWidget(self.octron.predict_video_predict_groupbox)
        self.octron.layoutWidget_3.setObjectName(u"layoutWidget_3")
        self.octron.layoutWidget_3.setGeometry(QRect(11, 232, 381, 37))
        self.octron.predict_finish_time_layout = QHBoxLayout(self.octron.layoutWidget_3)
        self.octron.predict_finish_time_layout.setObjectName(u"predict_finish_time_layout")
        self.octron.predict_finish_time_layout.setContentsMargins(0, 0, 1, 0)
        self.octron.predict_finish_time_label = QLabel(self.octron.layoutWidget_3)
        self.octron.predict_finish_time_label.setObjectName(u"predict_finish_time_label")
        self.octron.predict_finish_time_label.setEnabled(False)
        self.octron.predict_finish_time_label.setMinimumSize(QSize(0, 25))
        self.octron.predict_finish_time_label.setMaximumSize(QSize(16777215, 25))

        self.octron.predict_finish_time_layout.addWidget(self.octron.predict_finish_time_label)

        self.octron.predict_start_btn = QPushButton(self.octron.layoutWidget_3)
        self.octron.predict_start_btn.setObjectName(u"predict_start_btn")
        self.octron.predict_start_btn.setMinimumSize(QSize(90, 25))
        self.octron.predict_start_btn.setMaximumSize(QSize(90, 25))

        self.octron.predict_finish_time_layout.addWidget(self.octron.predict_start_btn)

        self.octron.layoutWidget8 = QWidget(self.octron.predict_video_predict_groupbox)
        self.octron.layoutWidget8.setObjectName(u"layoutWidget8")
        self.octron.layoutWidget8.setGeometry(QRect(10, 30, 381, 86))
        self.octron.gridLayout_2 = QGridLayout(self.octron.layoutWidget8)
        self.octron.gridLayout_2.setObjectName(u"gridLayout_2")
        self.octron.gridLayout_2.setContentsMargins(0, 0, 0, 10)
        self.octron.yolomodel_trained_list = QComboBox(self.octron.layoutWidget8)
        self.octron.yolomodel_trained_list.addItem("")
        self.octron.yolomodel_trained_list.setObjectName(u"yolomodel_trained_list")
        self.octron.yolomodel_trained_list.setEnabled(True)
        self.octron.yolomodel_trained_list.setMinimumSize(QSize(150, 25))
        self.octron.yolomodel_trained_list.setMaximumSize(QSize(150, 25))

        self.octron.gridLayout_2.addWidget(self.octron.yolomodel_trained_list, 0, 0, 2, 1, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop)

        self.octron.yolomodel_tracker_list = QComboBox(self.octron.layoutWidget8)
        self.octron.yolomodel_tracker_list.addItem("")
        self.octron.yolomodel_tracker_list.addItem("")
        self.octron.yolomodel_tracker_list.addItem("")
        self.octron.yolomodel_tracker_list.setObjectName(u"yolomodel_tracker_list")
        self.octron.yolomodel_tracker_list.setMinimumSize(QSize(110, 25))
        self.octron.yolomodel_tracker_list.setMaximumSize(QSize(110, 25))

        self.octron.gridLayout_2.addWidget(self.octron.yolomodel_tracker_list, 0, 1, 2, 1, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop)

        self.octron.open_when_finish_checkBox = QCheckBox(self.octron.layoutWidget8)
        self.octron.open_when_finish_checkBox.setObjectName(u"open_when_finish_checkBox")
        self.octron.open_when_finish_checkBox.setMinimumSize(QSize(90, 25))
        self.octron.open_when_finish_checkBox.setMaximumSize(QSize(100, 25))
        self.octron.open_when_finish_checkBox.setChecked(True)

        self.octron.gridLayout_2.addWidget(self.octron.open_when_finish_checkBox, 0, 2, 1, 1, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop)

        self.octron.single_subject_checkBox = QCheckBox(self.octron.layoutWidget8)
        self.octron.single_subject_checkBox.setObjectName(u"single_subject_checkBox")
        self.octron.single_subject_checkBox.setEnabled(True)
        self.octron.single_subject_checkBox.setMinimumSize(QSize(90, 25))
        self.octron.single_subject_checkBox.setMaximumSize(QSize(100, 25))
        self.octron.single_subject_checkBox.setChecked(False)

        self.octron.gridLayout_2.addWidget(self.octron.single_subject_checkBox, 1, 2, 1, 1, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop)

        self.octron.videos_for_prediction_list = QComboBox(self.octron.layoutWidget8)
        self.octron.videos_for_prediction_list.addItem("")
        self.octron.videos_for_prediction_list.addItem("")
        self.octron.videos_for_prediction_list.setObjectName(u"videos_for_prediction_list")
        self.octron.videos_for_prediction_list.setMinimumSize(QSize(270, 25))
        self.octron.videos_for_prediction_list.setMaximumSize(QSize(270, 25))
        self.octron.videos_for_prediction_list.setEditable(False)
        self.octron.videos_for_prediction_list.setMaxCount(15)
        self.octron.videos_for_prediction_list.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.octron.videos_for_prediction_list.setIconSize(QSize(14, 14))
        self.octron.videos_for_prediction_list.setFrame(False)

        self.octron.gridLayout_2.addWidget(self.octron.videos_for_prediction_list, 2, 0, 1, 2, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop)

        self.octron.overwrite_prediction_checkBox = QCheckBox(self.octron.layoutWidget8)
        self.octron.overwrite_prediction_checkBox.setObjectName(u"overwrite_prediction_checkBox")
        self.octron.overwrite_prediction_checkBox.setEnabled(True)
        self.octron.overwrite_prediction_checkBox.setMinimumSize(QSize(90, 25))
        self.octron.overwrite_prediction_checkBox.setMaximumSize(QSize(100, 25))
        self.octron.overwrite_prediction_checkBox.setChecked(False)

        self.octron.gridLayout_2.addWidget(self.octron.overwrite_prediction_checkBox, 2, 2, 1, 1, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop)

        self.octron.layoutWidget9 = QWidget(self.octron.predict_video_predict_groupbox)
        self.octron.layoutWidget9.setObjectName(u"layoutWidget9")
        self.octron.layoutWidget9.setGeometry(QRect(10, 120, 381, 62))
        self.octron.gridLayout_3 = QGridLayout(self.octron.layoutWidget9)
        self.octron.gridLayout_3.setObjectName(u"gridLayout_3")
        self.octron.gridLayout_3.setVerticalSpacing(0)
        self.octron.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.octron.prediction_mask_opening_label = QLabel(self.octron.layoutWidget9)
        self.octron.prediction_mask_opening_label.setObjectName(u"prediction_mask_opening_label")
        sizePolicy5.setHeightForWidth(self.octron.prediction_mask_opening_label.sizePolicy().hasHeightForWidth())
        self.octron.prediction_mask_opening_label.setSizePolicy(sizePolicy5)
        self.octron.prediction_mask_opening_label.setMinimumSize(QSize(75, 0))
        self.octron.prediction_mask_opening_label.setMaximumSize(QSize(40, 25))

        self.octron.gridLayout_3.addWidget(self.octron.prediction_mask_opening_label, 0, 0, 1, 1, Qt.AlignmentFlag.AlignLeft)

        self.octron.predict_mask_opening_spinbox = QDoubleSpinBox(self.octron.layoutWidget9)
        self.octron.predict_mask_opening_spinbox.setObjectName(u"predict_mask_opening_spinbox")
        self.octron.predict_mask_opening_spinbox.setMinimumSize(QSize(70, 25))
        self.octron.predict_mask_opening_spinbox.setMaximumSize(QSize(70, 25))
        self.octron.predict_mask_opening_spinbox.setDecimals(1)
        self.octron.predict_mask_opening_spinbox.setMaximum(5.000000000000000)
        self.octron.predict_mask_opening_spinbox.setSingleStep(0.250000000000000)
        self.octron.predict_mask_opening_spinbox.setValue(2.000000000000000)

        self.octron.gridLayout_3.addWidget(self.octron.predict_mask_opening_spinbox, 0, 1, 1, 1)

        self.octron.prediction_iou_label = QLabel(self.octron.layoutWidget9)
        self.octron.prediction_iou_label.setObjectName(u"prediction_iou_label")
        sizePolicy5.setHeightForWidth(self.octron.prediction_iou_label.sizePolicy().hasHeightForWidth())
        self.octron.prediction_iou_label.setSizePolicy(sizePolicy5)
        self.octron.prediction_iou_label.setMinimumSize(QSize(75, 0))
        self.octron.prediction_iou_label.setMaximumSize(QSize(75, 25))
        self.octron.prediction_iou_label.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.octron.gridLayout_3.addWidget(self.octron.prediction_iou_label, 0, 2, 1, 1, Qt.AlignmentFlag.AlignRight)

        self.octron.predict_iou_thresh_spinbox = QDoubleSpinBox(self.octron.layoutWidget9)
        self.octron.predict_iou_thresh_spinbox.setObjectName(u"predict_iou_thresh_spinbox")
        self.octron.predict_iou_thresh_spinbox.setMinimumSize(QSize(70, 25))
        self.octron.predict_iou_thresh_spinbox.setMaximumSize(QSize(70, 25))
        self.octron.predict_iou_thresh_spinbox.setMaximum(1.000000000000000)
        self.octron.predict_iou_thresh_spinbox.setSingleStep(0.100000000000000)
        self.octron.predict_iou_thresh_spinbox.setValue(0.400000000000000)

        self.octron.gridLayout_3.addWidget(self.octron.predict_iou_thresh_spinbox, 0, 3, 1, 1)

        self.octron.prediction_conf_thresh_label = QLabel(self.octron.layoutWidget9)
        self.octron.prediction_conf_thresh_label.setObjectName(u"prediction_conf_thresh_label")
        sizePolicy5.setHeightForWidth(self.octron.prediction_conf_thresh_label.sizePolicy().hasHeightForWidth())
        self.octron.prediction_conf_thresh_label.setSizePolicy(sizePolicy5)
        self.octron.prediction_conf_thresh_label.setMinimumSize(QSize(75, 0))
        self.octron.prediction_conf_thresh_label.setMaximumSize(QSize(40, 25))

        self.octron.gridLayout_3.addWidget(self.octron.prediction_conf_thresh_label, 1, 0, 1, 1, Qt.AlignmentFlag.AlignLeft)

        self.octron.predict_conf_thresh_spinbox = QDoubleSpinBox(self.octron.layoutWidget9)
        self.octron.predict_conf_thresh_spinbox.setObjectName(u"predict_conf_thresh_spinbox")
        self.octron.predict_conf_thresh_spinbox.setMinimumSize(QSize(70, 25))
        self.octron.predict_conf_thresh_spinbox.setMaximumSize(QSize(70, 25))
        self.octron.predict_conf_thresh_spinbox.setMaximum(1.000000000000000)
        self.octron.predict_conf_thresh_spinbox.setSingleStep(0.050000000000000)
        self.octron.predict_conf_thresh_spinbox.setValue(0.600000000000000)

        self.octron.gridLayout_3.addWidget(self.octron.predict_conf_thresh_spinbox, 1, 1, 1, 1)

        self.octron.prediction_skip_label = QLabel(self.octron.layoutWidget9)
        self.octron.prediction_skip_label.setObjectName(u"prediction_skip_label")
        sizePolicy5.setHeightForWidth(self.octron.prediction_skip_label.sizePolicy().hasHeightForWidth())
        self.octron.prediction_skip_label.setSizePolicy(sizePolicy5)
        self.octron.prediction_skip_label.setMinimumSize(QSize(75, 25))
        self.octron.prediction_skip_label.setMaximumSize(QSize(75, 25))
        self.octron.prediction_skip_label.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.octron.gridLayout_3.addWidget(self.octron.prediction_skip_label, 1, 2, 1, 1, Qt.AlignmentFlag.AlignRight)

        self.octron.skip_frames_analysis_spinBox = QSpinBox(self.octron.layoutWidget9)
        self.octron.skip_frames_analysis_spinBox.setObjectName(u"skip_frames_analysis_spinBox")
        self.octron.skip_frames_analysis_spinBox.setMinimumSize(QSize(70, 25))
        self.octron.skip_frames_analysis_spinBox.setMaximumSize(QSize(70, 25))
        self.octron.skip_frames_analysis_spinBox.setMaximum(1000)

        self.octron.gridLayout_3.addWidget(self.octron.skip_frames_analysis_spinBox, 1, 3, 1, 1)


        self.octron.predict_verticalLayout.addWidget(self.octron.predict_video_predict_groupbox)

        icon3 = QIcon()
        icon3.addFile(f"{base_path}/qt_gui/icons/noun-conversion-7158876.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.octron.toolBox.addItem(self.octron.predict_tab, icon3, u"Analyze (new) videos")

        self.octron.mainLayout.addWidget(self.octron.toolBox)

        self.octron.toolBox.raise_()
        self.octron.octron_logo.raise_()

        self.octron.toolBox.setCurrentIndex(0)
        self.octron.toolBox.layout().setSpacing(10)


    # setupUi
        self.octron.setWindowTitle(QCoreApplication.translate("self", u"octron_gui", None))
        self.octron.octron_logo.setText("")
        self.octron.folder_sect_groupbox.setTitle(QCoreApplication.translate("self", u"Project folder", None))
        self.octron.create_project_btn.setText(QCoreApplication.translate("self", u"\u2295 Choose", None))
        self.octron.project_folder_path_label.setText(QCoreApplication.translate("self", u"Project folder path", None))
        self.octron.project_video_drop_groupbox.setTitle(QCoreApplication.translate("self", u"Add new video file", None))
#if QT_CONFIG(tooltip)
        self.octron.video_file_drop_widget.setToolTip(QCoreApplication.translate("self", u"Drag and drop one .mp4 file here", None))
#endif // QT_CONFIG(tooltip)
        self.octron.project_existing_data_groupbox.setTitle(QCoreApplication.translate("self", u"Existing data", None))
        self.octron.toolBox.setItemText(self.octron.toolBox.indexOf(self.octron.project_tab), QCoreApplication.translate("self", u"Manage project", None))
#if QT_CONFIG(tooltip)
        self.octron.toolBox.setItemToolTip(self.octron.toolBox.indexOf(self.octron.project_tab), QCoreApplication.translate("self", u"Create new octron projects or load existing ones", None))
#endif // QT_CONFIG(tooltip)
        self.octron.horizontalGroupBox.setTitle(QCoreApplication.translate("self", u"Model selection", None))
        self.octron.sam2model_list.setItemText(0, QCoreApplication.translate("self", u"Choose model ...", None))

        self.octron.load_sam2model_btn.setText(QCoreApplication.translate("self", u"Load model", None))
        self.octron.annotate_layer_create_groupbox.setTitle(QCoreApplication.translate("self", u"Label manager", None))
#if QT_CONFIG(tooltip)
        self.octron.create_projection_layer_btn.setToolTip(QCoreApplication.translate("self", u"Create an average projection out of all segmented images for the current label", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(statustip)
        self.octron.create_projection_layer_btn.setStatusTip("")
#endif // QT_CONFIG(statustip)
        self.octron.create_projection_layer_btn.setText(QCoreApplication.translate("self", u"Visualize all", None))
        self.octron.layer_type_combobox.setItemText(0, QCoreApplication.translate("self", u"Type ... ", None))
        self.octron.layer_type_combobox.setItemText(1, QCoreApplication.translate("self", u"Shapes", None))
        self.octron.layer_type_combobox.setItemText(2, QCoreApplication.translate("self", u"Points", None))
        self.octron.layer_type_combobox.setItemText(3, QCoreApplication.translate("self", u"Anchors", None))

        self.octron.layer_type_combobox.setCurrentText(QCoreApplication.translate("self", u"Type ... ", None))
#if QT_CONFIG(tooltip)
        self.octron.label_suffix_lineedit.setToolTip(QCoreApplication.translate("self", u"The suffix disambiguates label layers from each other\n"
"that have the same label name.\n"
"For example:\n"
"The label could be octo and suffix 1 for the first octopus,\n"
"and octo and suffix 2 for the second octo ", None))
#endif // QT_CONFIG(tooltip)
        self.octron.label_suffix_lineedit.setPlaceholderText(QCoreApplication.translate("self", u"Suffix", None))
        self.octron.label_list_combobox.setItemText(0, QCoreApplication.translate("self", u"Label ... ", None))
        self.octron.label_list_combobox.setItemText(1, QCoreApplication.translate("self", u"\u2295 Create", None))
        self.octron.label_list_combobox.setItemText(2, QCoreApplication.translate("self", u"\u2296 Remove", None))

#if QT_CONFIG(tooltip)
        self.octron.label_list_combobox.setToolTip(QCoreApplication.translate("self", u"Select, add or remove labels", None))
#endif // QT_CONFIG(tooltip)
        self.octron.label_list_combobox.setCurrentText(QCoreApplication.translate("self", u"Label ... ", None))
#if QT_CONFIG(tooltip)
        self.octron.hard_reset_layer_btn.setToolTip(QCoreApplication.translate("self", u"Hard reset of the SAM2 predictor. Use this if prediction really did not go well for your data.", None))
#endif // QT_CONFIG(tooltip)
        self.octron.hard_reset_layer_btn.setText(QCoreApplication.translate("self", u"\u3004 Reset", None))
        self.octron.create_annotation_layer_btn.setText(QCoreApplication.translate("self", u"\u2295 Create", None))
        self.octron.annotate_layer_timeline_groupbox.setTitle(QCoreApplication.translate("self", u"Timeline control", None))
#if QT_CONFIG(tooltip)
        self.octron.annotation_jump_previous_btn.setToolTip(QCoreApplication.translate("self", u"Jump to last annotated frame", None))
#endif // QT_CONFIG(tooltip)
        self.octron.annotation_jump_previous_btn.setText(QCoreApplication.translate("self", u"\u226a Jump to previous", None))
#if QT_CONFIG(tooltip)
        self.octron.annotation_jump_next_btn.setToolTip(QCoreApplication.translate("self", u"Jump to next annotated frame", None))
#endif // QT_CONFIG(tooltip)
        self.octron.annotation_jump_next_btn.setText(QCoreApplication.translate("self", u"Jump to next \u226b", None))
        self.octron.annotate_layer_predict_groupbox.setTitle(QCoreApplication.translate("self", u"Batch prediction", None))
#if QT_CONFIG(tooltip)
        self.octron.batch_predict_progressbar.setToolTip(QCoreApplication.translate("self", u"<html><head/><body><p>Batch predict progress bar</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.octron.batch_predict_progressbar.setFormat(QCoreApplication.translate("self", u"%p%", None))
        self.octron.skip_label.setText(QCoreApplication.translate("self", u"Skip", None))
#if QT_CONFIG(tooltip)
        self.octron.skip_frames_spinbox.setToolTip(QCoreApplication.translate("self", u"How many frames should be skipped in\n"
"batch prediction?", None))
#endif // QT_CONFIG(tooltip)
        self.octron.skip_frames_spinbox.setSuffix("")
        self.octron.skip_frames_spinbox.setPrefix("")
#if QT_CONFIG(tooltip)
        self.octron.predict_next_oneframe_btn.setToolTip(QCoreApplication.translate("self", u"Predict next frame", None))
#endif // QT_CONFIG(tooltip)
        self.octron.predict_next_oneframe_btn.setText("")
#if QT_CONFIG(tooltip)
        self.octron.predict_next_batch_btn.setToolTip(QCoreApplication.translate("self", u"Predict batch of next frames", None))
#endif // QT_CONFIG(tooltip)
        self.octron.predict_next_batch_btn.setText("")
        self.octron.toolBox.setItemText(self.octron.toolBox.indexOf(self.octron.annotate_tab), QCoreApplication.translate("self", u"Generate annotation data", None))
#if QT_CONFIG(tooltip)
        self.octron.toolBox.setItemToolTip(self.octron.toolBox.indexOf(self.octron.annotate_tab), QCoreApplication.translate("self", u"Create annotation data for training, i.e. add segmentation or keypoint data on videos.", None))
#endif // QT_CONFIG(tooltip)
        self.octron.train_generate_groupbox.setTitle(QCoreApplication.translate("self", u"Generate training data", None))
#if QT_CONFIG(tooltip)
        self.octron.train_polygons_overall_progressbar.setToolTip(QCoreApplication.translate("self", u"<html><head/><body><p>Batch predict progress bar</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.octron.train_polygons_overall_progressbar.setFormat(QCoreApplication.translate("self", u"%p%", None))
#if QT_CONFIG(tooltip)
        self.octron.train_polygons_frames_progressbar.setToolTip(QCoreApplication.translate("self", u"<html><head/><body><p>Batch predict progress bar</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.octron.train_polygons_frames_progressbar.setFormat(QCoreApplication.translate("self", u"%p%", None))
        self.octron.train_polygons_label.setText(QCoreApplication.translate("self", u"label", None))
#if QT_CONFIG(tooltip)
        self.octron.train_export_overall_progressbar.setToolTip(QCoreApplication.translate("self", u"<html><head/><body><p>Batch predict progress bar</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.octron.train_export_overall_progressbar.setFormat(QCoreApplication.translate("self", u"%p%", None))
#if QT_CONFIG(tooltip)
        self.octron.train_export_frames_progressbar.setToolTip(QCoreApplication.translate("self", u"<html><head/><body><p>Batch predict progress bar</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.octron.train_export_frames_progressbar.setFormat(QCoreApplication.translate("self", u"%p%", None))
        self.octron.train_export_label.setText(QCoreApplication.translate("self", u"label and split", None))
#if QT_CONFIG(tooltip)
        self.octron.train_prune_checkBox.setToolTip(QCoreApplication.translate("self", u"Exclude frames in which only a subset of all\n"
"labels is present.", None))
#endif // QT_CONFIG(tooltip)
        self.octron.train_prune_checkBox.setText(QCoreApplication.translate("self", u"Prune", None))
#if QT_CONFIG(tooltip)
        self.octron.train_data_watershed_checkBox.setToolTip(QCoreApplication.translate("self", u"Enable watershed on mask data. This helps to separate masks that \n"
"are on the same layer and carry the same label assignment,\n"
"but should be separate entities in the training data.", None))
#endif // QT_CONFIG(tooltip)
        self.octron.train_data_watershed_checkBox.setText(QCoreApplication.translate("self", u"Watershed", None))
        self.octron.train_data_overwrite_checkBox.setText(QCoreApplication.translate("self", u"Overwrite", None))
        self.octron.training_data_folder_label.setText("")
        self.octron.generate_training_data_btn.setText(QCoreApplication.translate("self", u"Generate", None))
        self.octron.train_train_groupbox.setTitle(QCoreApplication.translate("self", u"Train", None))
        self.octron.num_epochs_label.setText(QCoreApplication.translate("self", u"Epochs", None))
        self.octron.save_period_label.setText(QCoreApplication.translate("self", u"Save period", None))
#if QT_CONFIG(tooltip)
        self.octron.num_epochs_input.setToolTip(QCoreApplication.translate("self", u"How many epochs in total\n"
"should be trained?\n"
"Recommended are at least ~50.", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.octron.save_period_input.setToolTip(QCoreApplication.translate("self", u"After how many epochs should\n"
"(intermediary) output models be saved?", None))
#endif // QT_CONFIG(tooltip)
        self.octron.train_resume_checkBox.setText(QCoreApplication.translate("self", u"Resume", None))
        self.octron.train_training_overwrite_checkBox.setText(QCoreApplication.translate("self", u"Overwrite", None))
#if QT_CONFIG(tooltip)
        self.octron.launch_tensorboard_checkBox.setToolTip(QCoreApplication.translate("self", u"Start tensorboard (open browser window)", None))
#endif // QT_CONFIG(tooltip)
        self.octron.launch_tensorboard_checkBox.setText(QCoreApplication.translate("self", u"Tensorbrd", None))
        self.octron.yolomodel_list.setItemText(0, QCoreApplication.translate("self", u"Choose model ...", None))

        self.octron.yoloimagesize_list.setItemText(0, QCoreApplication.translate("self", u"Img. size", None))
        self.octron.yoloimagesize_list.setItemText(1, QCoreApplication.translate("self", u"640", None))
        self.octron.yoloimagesize_list.setItemText(2, QCoreApplication.translate("self", u"1024", None))

#if QT_CONFIG(tooltip)
        self.octron.yoloimagesize_list.setToolTip(QCoreApplication.translate("self", u"Image size used for training within YOLO", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.octron.train_epochs_progressbar.setToolTip(QCoreApplication.translate("self", u"<html><head/><body><p>Batch predict progress bar</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.octron.train_epochs_progressbar.setFormat(QCoreApplication.translate("self", u"%p%", None))
#if QT_CONFIG(tooltip)
        self.octron.train_finishtime_label.setToolTip(QCoreApplication.translate("self", u"Approximate time of training finish", None))
#endif // QT_CONFIG(tooltip)
        self.octron.train_finishtime_label.setText(QCoreApplication.translate("self", u"Finish time", None))
        self.octron.start_stop_training_btn.setText(QCoreApplication.translate("self", u"Start", None))
        self.octron.toolBox.setItemText(self.octron.toolBox.indexOf(self.octron.train_tab), QCoreApplication.translate("self", u"Train model", None))
#if QT_CONFIG(tooltip)
        self.octron.toolBox.setItemToolTip(self.octron.toolBox.indexOf(self.octron.train_tab), QCoreApplication.translate("self", u"Train a new or existing model with generated training data", None))
#endif // QT_CONFIG(tooltip)
        self.octron.predict_video_drop_groupbox.setTitle(QCoreApplication.translate("self", u"Add video files", None))
#if QT_CONFIG(tooltip)
        self.octron.predict_video_drop_widget.setToolTip(QCoreApplication.translate("self", u"Drag and drop .mp4 files here", None))
#endif // QT_CONFIG(tooltip)
        self.octron.predict_video_predict_groupbox.setTitle(QCoreApplication.translate("self", u"Create predictions from videos", None))
#if QT_CONFIG(tooltip)
        self.octron.predict_overall_progressbar.setToolTip(QCoreApplication.translate("self", u"<html><head/><body><p>Batch predict progress bar</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.octron.predict_overall_progressbar.setFormat(QCoreApplication.translate("self", u"%p%", None))
#if QT_CONFIG(tooltip)
        self.octron.predict_current_video_progressbar.setToolTip(QCoreApplication.translate("self", u"<html><head/><body><p>Batch predict progress bar</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.octron.predict_current_video_progressbar.setFormat(QCoreApplication.translate("self", u"%p%", None))
        self.octron.predict_current_videoname_label.setText(QCoreApplication.translate("self", u"video name", None))
        self.octron.predict_finish_time_label.setText(QCoreApplication.translate("self", u"Current video finishes in:", None))
#if QT_CONFIG(tooltip)
        self.octron.predict_start_btn.setToolTip(QCoreApplication.translate("self", u"Yeah!", None))
#endif // QT_CONFIG(tooltip)
        self.octron.predict_start_btn.setText(QCoreApplication.translate("self", u"Let's go!", None))
        self.octron.yolomodel_trained_list.setItemText(0, QCoreApplication.translate("self", u"Choose model ...", None))

#if QT_CONFIG(tooltip)
        self.octron.yolomodel_trained_list.setToolTip(QCoreApplication.translate("self", u"OCTRON user trained models that are found in the project path", None))
#endif // QT_CONFIG(tooltip)
        self.octron.yolomodel_tracker_list.setItemText(0, QCoreApplication.translate("self", u"Tracker ...", None))
        self.octron.yolomodel_tracker_list.setItemText(1, QCoreApplication.translate("self", u"BoTSORT", None))
        self.octron.yolomodel_tracker_list.setItemText(2, QCoreApplication.translate("self", u"ByteTrack", None))

#if QT_CONFIG(tooltip)
        self.octron.yolomodel_tracker_list.setToolTip(QCoreApplication.translate("self", u"OCTRON user trained models that are found in the project path", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.octron.open_when_finish_checkBox.setToolTip(QCoreApplication.translate("self", u"Open the resuts in new napari window when finished", None))
#endif // QT_CONFIG(tooltip)
        self.octron.open_when_finish_checkBox.setText(QCoreApplication.translate("self", u"View results", None))
#if QT_CONFIG(tooltip)
        self.octron.single_subject_checkBox.setToolTip(QCoreApplication.translate("self", u"Click this if you expect only one subject to be tracked per label", None))
#endif // QT_CONFIG(tooltip)
        self.octron.single_subject_checkBox.setText(QCoreApplication.translate("self", u"1 subject", None))
        self.octron.videos_for_prediction_list.setItemText(0, QCoreApplication.translate("self", u"Videos", None))
        self.octron.videos_for_prediction_list.setItemText(1, QCoreApplication.translate("self", u"\u2296 Remove", None))

#if QT_CONFIG(tooltip)
        self.octron.videos_for_prediction_list.setToolTip(QCoreApplication.translate("self", u"Select, add or remove labels", None))
#endif // QT_CONFIG(tooltip)
        self.octron.videos_for_prediction_list.setCurrentText(QCoreApplication.translate("self", u"Videos", None))
#if QT_CONFIG(tooltip)
        self.octron.overwrite_prediction_checkBox.setToolTip(QCoreApplication.translate("self", u"Overwrite previous analysis results? ", None))
#endif // QT_CONFIG(tooltip)
        self.octron.overwrite_prediction_checkBox.setText(QCoreApplication.translate("self", u"Overwrite", None))
#if QT_CONFIG(tooltip)
        self.octron.prediction_mask_opening_label.setToolTip(QCoreApplication.translate("self", u"Morphological opening of predicted masks.\n"
"This gets rid of some noise. 2 is a good value to start with.", None))
#endif // QT_CONFIG(tooltip)
        self.octron.prediction_mask_opening_label.setText(QCoreApplication.translate("self", u"Opening", None))
#if QT_CONFIG(tooltip)
        self.octron.prediction_iou_label.setToolTip(QCoreApplication.translate("self", u"Intersection over union. This threshold determines how much overlap between bounding boxes\n"
"is allowed before they are considered to be detecting the same object.\n"
"At IOU=0 all detected objects > conf. thresh\n"
"of one label will be fused into one mask.", None))
#endif // QT_CONFIG(tooltip)
        self.octron.prediction_iou_label.setText(QCoreApplication.translate("self", u"IOU", None))
#if QT_CONFIG(tooltip)
        self.octron.prediction_conf_thresh_label.setToolTip(QCoreApplication.translate("self", u"Confidence threshold for accepting prediction results as real", None))
#endif // QT_CONFIG(tooltip)
        self.octron.prediction_conf_thresh_label.setText(QCoreApplication.translate("self", u"Confidence", None))
#if QT_CONFIG(tooltip)
        self.octron.prediction_skip_label.setToolTip(QCoreApplication.translate("self", u"Skip frames in videos? 0: All frames are analyzed, >0: This many frames are being skipped. ", None))
#endif // QT_CONFIG(tooltip)
        self.octron.prediction_skip_label.setText(QCoreApplication.translate("self", u"Skip frames", None))
        self.octron.toolBox.setItemText(self.octron.toolBox.indexOf(self.octron.predict_tab), QCoreApplication.translate("self", u"Analyze (new) videos", None))
#if QT_CONFIG(tooltip)
        self.octron.toolBox.setItemToolTip(self.octron.toolBox.indexOf(self.octron.predict_tab), QCoreApplication.translate("self", u"Use trained models to run predictions on new videos", None))
#endif // QT_CONFIG(tooltip)
    # retranslateUi

