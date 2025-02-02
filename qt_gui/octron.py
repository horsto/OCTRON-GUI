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
        octron_widgetui.resize(410, 655)
        octron_widgetui.setMinimumSize(QSize(400, 0))
        self.verticalLayoutWidget = QWidget(octron_widgetui)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QRect(0, 0, 502, 611))
        self.mainLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.mainLayout.setSpacing(0)
        self.mainLayout.setObjectName(u"mainLayout")
        self.mainLayout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.octron_logo = QLabel(self.verticalLayoutWidget)
        self.octron_logo.setObjectName(u"octron_logo")
        self.octron_logo.setEnabled(True)
        self.octron_logo.setPixmap(QPixmap(u"octron_logo.svg"))
        self.octron_logo.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop)

        self.mainLayout.addWidget(self.octron_logo)

        self.toolBox = QToolBox(self.verticalLayoutWidget)
        self.toolBox.setObjectName(u"toolBox")
        self.toolBox.setCursor(QCursor(Qt.ArrowCursor))
        self.project_tab = QWidget()
        self.project_tab.setObjectName(u"project_tab")
        self.project_tab.setGeometry(QRect(0, 0, 500, 334))
        self.verticalLayoutWidget_2 = QWidget(self.project_tab)
        self.verticalLayoutWidget_2.setObjectName(u"verticalLayoutWidget_2")
        self.verticalLayoutWidget_2.setGeometry(QRect(0, 0, 381, 171))
        self.project_tab_vertical_layout = QVBoxLayout(self.verticalLayoutWidget_2)
        self.project_tab_vertical_layout.setObjectName(u"project_tab_vertical_layout")
        self.project_tab_vertical_layout.setContentsMargins(0, 0, 0, 0)
        self.create_project_button = QPushButton(self.verticalLayoutWidget_2)
        self.create_project_button.setObjectName(u"create_project_button")
        self.create_project_button.setMouseTracking(False)
        self.create_project_button.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        icon = QIcon(QIcon.fromTheme(u"QIcon::ThemeIcon::ListAdd"))
        self.create_project_button.setIcon(icon)

        self.project_tab_vertical_layout.addWidget(self.create_project_button, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop)

        icon1 = QIcon(QIcon.fromTheme(u"QIcon::ThemeIcon::DocumentOpen"))
        self.toolBox.addItem(self.project_tab, icon1, u"Project")
        self.annotate_tab = QWidget()
        self.annotate_tab.setObjectName(u"annotate_tab")
        self.annotate_tab.setGeometry(QRect(0, 0, 500, 334))
        icon2 = QIcon(QIcon.fromTheme(u"QIcon::ThemeIcon::MediaPlaybackStop"))
        self.toolBox.addItem(self.annotate_tab, icon2, u"Create training data")
        self.train_tab = QWidget()
        self.train_tab.setObjectName(u"train_tab")
        self.train_tab.setGeometry(QRect(0, 0, 500, 334))
        icon3 = QIcon(QIcon.fromTheme(u"QIcon::ThemeIcon::Computer"))
        self.toolBox.addItem(self.train_tab, icon3, u"Train model")
        self.predict_tab = QWidget()
        self.predict_tab.setObjectName(u"predict_tab")
        self.predict_tab.setGeometry(QRect(0, 0, 500, 334))
        icon4 = QIcon(QIcon.fromTheme(u"QIcon::ThemeIcon::ViewFullscreen"))
        self.toolBox.addItem(self.predict_tab, icon4, u"Predict on new videos")

        self.mainLayout.addWidget(self.toolBox)


        self.retranslateUi(octron_widgetui)

        self.toolBox.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(octron_widgetui)
    # setupUi

    def retranslateUi(self, octron_widgetui):
        octron_widgetui.setWindowTitle(QCoreApplication.translate("octron_widgetui", u"octron_gui", None))
        self.octron_logo.setText("")
        self.create_project_button.setText(QCoreApplication.translate("octron_widgetui", u"Create Project", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.project_tab), QCoreApplication.translate("octron_widgetui", u"Project", None))
#if QT_CONFIG(tooltip)
        self.toolBox.setItemToolTip(self.toolBox.indexOf(self.project_tab), QCoreApplication.translate("octron_widgetui", u"Create new octron projects or load existing ones", None))
#endif // QT_CONFIG(tooltip)
        self.toolBox.setItemText(self.toolBox.indexOf(self.annotate_tab), QCoreApplication.translate("octron_widgetui", u"Create training data", None))
#if QT_CONFIG(tooltip)
        self.toolBox.setItemToolTip(self.toolBox.indexOf(self.annotate_tab), QCoreApplication.translate("octron_widgetui", u"Create segmentation data that can be used to train models ", None))
#endif // QT_CONFIG(tooltip)
        self.toolBox.setItemText(self.toolBox.indexOf(self.train_tab), QCoreApplication.translate("octron_widgetui", u"Train model", None))
#if QT_CONFIG(tooltip)
        self.toolBox.setItemToolTip(self.toolBox.indexOf(self.train_tab), QCoreApplication.translate("octron_widgetui", u"Train model with generated training data", None))
#endif // QT_CONFIG(tooltip)
        self.toolBox.setItemText(self.toolBox.indexOf(self.predict_tab), QCoreApplication.translate("octron_widgetui", u"Predict on new videos", None))
#if QT_CONFIG(tooltip)
        self.toolBox.setItemToolTip(self.toolBox.indexOf(self.predict_tab), QCoreApplication.translate("octron_widgetui", u"Use trained models to run predictions on new videos", None))
#endif // QT_CONFIG(tooltip)
    # retranslateUi

