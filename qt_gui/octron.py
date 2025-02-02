# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'octron.ui'
##
## Created by: Qt User Interface Compiler version 5.15.16
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *  # type: ignore
from PySide2.QtGui import *  # type: ignore
from PySide2.QtWidgets import *  # type: ignore


class Ui_octron_widgetui(object):
    def setupUi(self, octron_widgetui):
        if not octron_widgetui.objectName():
            octron_widgetui.setObjectName(u"octron_widgetui")
        octron_widgetui.setEnabled(True)
        octron_widgetui.resize(410, 500)
        octron_widgetui.setMinimumSize(QSize(410, 500))
        octron_widgetui.setMaximumSize(QSize(410, 1000))
        octron_widgetui.setCursor(QCursor(Qt.ArrowCursor))
        octron_widgetui.setWindowOpacity(1.000000000000000)
        self.verticalLayoutWidget = QWidget(octron_widgetui)
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
        self.octron_logo.setPixmap(QPixmap(u"octron_logo.svg"))
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
        icon.addFile(u"icons/noun-project-7158867.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.toolBox.addItem(self.project_tab, icon, u"Project")
        self.annotate_tab = QWidget()
        self.annotate_tab.setObjectName(u"annotate_tab")
        self.annotate_tab.setGeometry(QRect(0, 0, 410, 224))
        sizePolicy1.setHeightForWidth(self.annotate_tab.sizePolicy().hasHeightForWidth())
        self.annotate_tab.setSizePolicy(sizePolicy1)
        icon1 = QIcon()
        icon1.addFile(u"icons/noun-copywriting-7158879.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.toolBox.addItem(self.annotate_tab, icon1, u"Generate training data (annotate)")
        self.train_tab = QWidget()
        self.train_tab.setObjectName(u"train_tab")
        self.train_tab.setGeometry(QRect(0, 0, 410, 224))
        sizePolicy1.setHeightForWidth(self.train_tab.sizePolicy().hasHeightForWidth())
        self.train_tab.setSizePolicy(sizePolicy1)
        icon2 = QIcon()
        icon2.addFile(u"icons/noun-rocket-7158872.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.toolBox.addItem(self.train_tab, icon2, u"Train model")
        self.predict_tab = QWidget()
        self.predict_tab.setObjectName(u"predict_tab")
        self.predict_tab.setGeometry(QRect(0, 0, 410, 224))
        sizePolicy1.setHeightForWidth(self.predict_tab.sizePolicy().hasHeightForWidth())
        self.predict_tab.setSizePolicy(sizePolicy1)
        icon3 = QIcon()
        icon3.addFile(u"icons/noun-conversion-7158876.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.toolBox.addItem(self.predict_tab, icon3, u"Analyze (new) videos")

        self.mainLayout.addWidget(self.toolBox)

        self.toolBox.raise_()
        self.octron_logo.raise_()

        self.retranslateUi(octron_widgetui)

        self.toolBox.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(octron_widgetui)
    # setupUi

    def retranslateUi(self, octron_widgetui):
        octron_widgetui.setWindowTitle(QCoreApplication.translate("octron_widgetui", u"octron_gui", None))
        self.octron_logo.setText("")
        self.toolBox.setItemText(self.toolBox.indexOf(self.project_tab), QCoreApplication.translate("octron_widgetui", u"Project", None))
#if QT_CONFIG(tooltip)
        self.toolBox.setItemToolTip(self.toolBox.indexOf(self.project_tab), QCoreApplication.translate("octron_widgetui", u"Create new octron projects or load existing ones", None))
#endif // QT_CONFIG(tooltip)
        self.toolBox.setItemText(self.toolBox.indexOf(self.annotate_tab), QCoreApplication.translate("octron_widgetui", u"Generate training data (annotate)", None))
#if QT_CONFIG(tooltip)
        self.toolBox.setItemToolTip(self.toolBox.indexOf(self.annotate_tab), QCoreApplication.translate("octron_widgetui", u"Create annotation data for training, i.e. add segmentation or keypoint data on videos.", None))
#endif // QT_CONFIG(tooltip)
        self.toolBox.setItemText(self.toolBox.indexOf(self.train_tab), QCoreApplication.translate("octron_widgetui", u"Train model", None))
#if QT_CONFIG(tooltip)
        self.toolBox.setItemToolTip(self.toolBox.indexOf(self.train_tab), QCoreApplication.translate("octron_widgetui", u"Train a new or existing model with generated training data", None))
#endif // QT_CONFIG(tooltip)
        self.toolBox.setItemText(self.toolBox.indexOf(self.predict_tab), QCoreApplication.translate("octron_widgetui", u"Analyze (new) videos", None))
#if QT_CONFIG(tooltip)
        self.toolBox.setItemToolTip(self.toolBox.indexOf(self.predict_tab), QCoreApplication.translate("octron_widgetui", u"Use trained models to run predictions on new videos", None))
#endif // QT_CONFIG(tooltip)
    # retranslateUi

