# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'octron.ui'
##
## Created by: Qt User Interface Compiler version 5.15.16
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from qtpy.QtCore import *  # type: ignore
from qtpy.QtGui import *  # type: ignore
from qtpy.QtWidgets import *  # type: ignore


class Ui_octron_widgetui(object):
    def setupUi(self):
        if not self.objectName():
            self.setObjectName(u"self")
        self.setEnabled(True)
        self.resize(410, 600)
        self.setMinimumSize(QSize(410, 500))
        self.setMaximumSize(QSize(410, 1000))
        self.setCursor(QCursor(Qt.ArrowCursor))
        self.setWindowOpacity(1.000000000000000)
        self.verticalLayoutWidget = QWidget(self)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QRect(0, 0, 412, 591))
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
        self.project_tab.setGeometry(QRect(0, 0, 410, 314))
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
        self.annotate_tab.setGeometry(QRect(0, 0, 410, 314))
        sizePolicy1.setHeightForWidth(self.annotate_tab.sizePolicy().hasHeightForWidth())
        self.annotate_tab.setSizePolicy(sizePolicy1)
        self.annotate_tab.setMaximumSize(QSize(410, 16777215))
        self.verticalLayoutWidget_2 = QWidget(self.annotate_tab)
        self.verticalLayoutWidget_2.setObjectName(u"verticalLayoutWidget_2")
        self.verticalLayoutWidget_2.setGeometry(QRect(0, 0, 406, 311))
        self.annotate_vertical_layout = QVBoxLayout(self.verticalLayoutWidget_2)
#ifndef Q_OS_MAC
        self.annotate_vertical_layout.setSpacing(-1)
#endif
        self.annotate_vertical_layout.setObjectName(u"annotate_vertical_layout")
        self.annotate_vertical_layout.setContentsMargins(0, 0, 0, 0)
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
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.sam2model_list = QComboBox(self.horizontalGroupBox)
        self.sam2model_list.addItem("")
        self.sam2model_list.setObjectName(u"sam2model_list")
        self.sam2model_list.setMinimumSize(QSize(167, 0))
        self.sam2model_list.setMaximumSize(QSize(167, 25))

        self.horizontalLayout_8.addWidget(self.sam2model_list, 0, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.load_model_btn = QPushButton(self.horizontalGroupBox)
        self.load_model_btn.setObjectName(u"load_model_btn")
        self.load_model_btn.setMaximumSize(QSize(250, 60))

        self.horizontalLayout_8.addWidget(self.load_model_btn, 0, Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)


        self.annotate_vertical_layout.addWidget(self.horizontalGroupBox)

        self.annotate_layer_create_groupbox = QGroupBox(self.verticalLayoutWidget_2)
        self.annotate_layer_create_groupbox.setObjectName(u"annotate_layer_create_groupbox")
        sizePolicy2.setHeightForWidth(self.annotate_layer_create_groupbox.sizePolicy().hasHeightForWidth())
        self.annotate_layer_create_groupbox.setSizePolicy(sizePolicy2)
        self.annotate_layer_create_groupbox.setMinimumSize(QSize(400, 60))
        self.annotate_layer_create_groupbox.setMaximumSize(QSize(400, 60))
        self.horizontalLayout_2 = QHBoxLayout(self.annotate_layer_create_groupbox)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.layer_type_combobox = QComboBox(self.annotate_layer_create_groupbox)
        self.layer_type_combobox.addItem("")
        self.layer_type_combobox.addItem("")
        self.layer_type_combobox.addItem("")
        self.layer_type_combobox.addItem("")
        self.layer_type_combobox.setObjectName(u"layer_type_combobox")
        self.layer_type_combobox.setMinimumSize(QSize(167, 0))
        self.layer_type_combobox.setMaximumSize(QSize(167, 25))
        self.layer_type_combobox.setMaxCount(15)
        self.layer_type_combobox.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.layer_type_combobox.setIconSize(QSize(14, 14))
        self.layer_type_combobox.setFrame(False)

        self.horizontalLayout_2.addWidget(self.layer_type_combobox, 0, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.label_name_lineedit = QLineEdit(self.annotate_layer_create_groupbox)
        self.label_name_lineedit.setObjectName(u"label_name_lineedit")
        self.label_name_lineedit.setMinimumSize(QSize(140, 26))
        self.label_name_lineedit.setMaximumSize(QSize(140, 26))
        self.label_name_lineedit.setInputMask(u"")
        self.label_name_lineedit.setText(u"")
        self.label_name_lineedit.setMaxLength(100)

        self.horizontalLayout_2.addWidget(self.label_name_lineedit, 0, Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)

        self.create_annotation_layer_btn = QPushButton(self.annotate_layer_create_groupbox)
        self.create_annotation_layer_btn.setObjectName(u"create_annotation_layer_btn")
        self.create_annotation_layer_btn.setMaximumSize(QSize(60, 60))

        self.horizontalLayout_2.addWidget(self.create_annotation_layer_btn, 0, Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)


        self.annotate_vertical_layout.addWidget(self.annotate_layer_create_groupbox)

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
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.kernel_label = QLabel(self.annotate_param_groupbox)
        self.kernel_label.setObjectName(u"kernel_label")
        self.kernel_label.setMaximumSize(QSize(400, 25))

        self.horizontalLayout_4.addWidget(self.kernel_label)

        self.opening_kernel_radius_input = QSpinBox(self.annotate_param_groupbox)
        self.opening_kernel_radius_input.setObjectName(u"opening_kernel_radius_input")
        self.opening_kernel_radius_input.setMinimumSize(QSize(60, 25))
        self.opening_kernel_radius_input.setMaximumSize(QSize(60, 25))
        self.opening_kernel_radius_input.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.horizontalLayout_4.addWidget(self.opening_kernel_radius_input)

        self.kernelpx_label = QLabel(self.annotate_param_groupbox)
        self.kernelpx_label.setObjectName(u"kernelpx_label")
        self.kernelpx_label.setMinimumSize(QSize(18, 25))
        self.kernelpx_label.setMaximumSize(QSize(18, 25))

        self.horizontalLayout_4.addWidget(self.kernelpx_label)


        self.annotate_vertical_layout.addWidget(self.annotate_param_groupbox, 0, Qt.AlignmentFlag.AlignBottom)

        self.annotate_layer_predict_groupbox = QGroupBox(self.verticalLayoutWidget_2)
        self.annotate_layer_predict_groupbox.setObjectName(u"annotate_layer_predict_groupbox")
        sizePolicy2.setHeightForWidth(self.annotate_layer_predict_groupbox.sizePolicy().hasHeightForWidth())
        self.annotate_layer_predict_groupbox.setSizePolicy(sizePolicy2)
        self.annotate_layer_predict_groupbox.setMinimumSize(QSize(400, 60))
        self.annotate_layer_predict_groupbox.setMaximumSize(QSize(400, 60))
        self.horizontalLayout_7 = QHBoxLayout(self.annotate_layer_predict_groupbox)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.batch_predict_progressbar = QProgressBar(self.annotate_layer_predict_groupbox)
        self.batch_predict_progressbar.setObjectName(u"batch_predict_progressbar")
        self.batch_predict_progressbar.setMinimumSize(QSize(0, 25))
        self.batch_predict_progressbar.setMaximumSize(QSize(250, 25))
        self.batch_predict_progressbar.setMaximum(20)
        self.batch_predict_progressbar.setValue(0)

        self.horizontalLayout_7.addWidget(self.batch_predict_progressbar, 0, Qt.AlignmentFlag.AlignVCenter)

        self.predict_next_batch_btn = QPushButton(self.annotate_layer_predict_groupbox)
        self.predict_next_batch_btn.setObjectName(u"predict_next_batch_btn")
        self.predict_next_batch_btn.setEnabled(False)
        self.predict_next_batch_btn.setMaximumSize(QSize(250, 60))

        self.horizontalLayout_7.addWidget(self.predict_next_batch_btn, 0, Qt.AlignmentFlag.AlignVCenter)


        self.annotate_vertical_layout.addWidget(self.annotate_layer_predict_groupbox, 0, Qt.AlignmentFlag.AlignBottom)

        icon1 = QIcon()
        icon1.addFile(u"qt_gui/icons/noun-copywriting-7158879.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.toolBox.addItem(self.annotate_tab, icon1, u"Generate training data (annotate)")
        self.train_tab = QWidget()
        self.train_tab.setObjectName(u"train_tab")
        self.train_tab.setGeometry(QRect(0, 0, 410, 314))
        sizePolicy1.setHeightForWidth(self.train_tab.sizePolicy().hasHeightForWidth())
        self.train_tab.setSizePolicy(sizePolicy1)
        icon2 = QIcon()
        icon2.addFile(u"qt_gui/icons/noun-rocket-7158872.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.toolBox.addItem(self.train_tab, icon2, u"Train model")
        self.predict_tab = QWidget()
        self.predict_tab.setObjectName(u"predict_tab")
        self.predict_tab.setGeometry(QRect(0, 0, 410, 314))
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
        self.horizontalGroupBox.setTitle(QCoreApplication.translate("self", u"Model selection", None))
        self.sam2model_list.setItemText(0, QCoreApplication.translate("self", u"Choose model ...", None))

        self.load_model_btn.setText(QCoreApplication.translate("self", u"Load model", None))
        self.annotate_layer_create_groupbox.setTitle(QCoreApplication.translate("self", u"Layer controls", None))
        self.layer_type_combobox.setItemText(0, QCoreApplication.translate("self", u"Layer Type", None))
        self.layer_type_combobox.setItemText(1, QCoreApplication.translate("self", u"Shape Layer", None))
        self.layer_type_combobox.setItemText(2, QCoreApplication.translate("self", u"Point->Mask Layer", None))
        self.layer_type_combobox.setItemText(3, QCoreApplication.translate("self", u"Anchor point Layer ", None))

        self.layer_type_combobox.setCurrentText(QCoreApplication.translate("self", u"Layer Type", None))
        self.label_name_lineedit.setPlaceholderText(QCoreApplication.translate("self", u"Label name", None))
        self.create_annotation_layer_btn.setText(QCoreApplication.translate("self", u"Create", None))
        self.annotate_param_groupbox.setTitle(QCoreApplication.translate("self", u"Parameters", None))
        self.kernel_label.setText(QCoreApplication.translate("self", u"Opening kernel radius", None))
        self.kernelpx_label.setText(QCoreApplication.translate("self", u"px", None))
        self.annotate_layer_predict_groupbox.setTitle(QCoreApplication.translate("self", u"Batch prediction", None))
#if QT_CONFIG(tooltip)
        self.batch_predict_progressbar.setToolTip(QCoreApplication.translate("self", u"<html><head/><body><p>Batch predict progress bar</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.batch_predict_progressbar.setFormat(QCoreApplication.translate("self", u"%p%", None))
        self.predict_next_batch_btn.setText(QCoreApplication.translate("self", u"Predict next 20 frames", None))
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

