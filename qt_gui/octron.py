# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'octron.ui'
##
## Created by: Qt User Interface Compiler version 5.15.13
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *  # type: ignore
from PySide2.QtGui import *  # type: ignore
from PySide2.QtWidgets import *  # type: ignore


class Ui_octron_gui(object):
    def setupUi(self, octron_gui):
        if not octron_gui.objectName():
            octron_gui.setObjectName(u"octron_gui")
        octron_gui.setEnabled(True)
        octron_gui.resize(388, 655)
        self.tabs = QTabWidget(octron_gui)
        self.tabs.setObjectName(u"tabs")
        self.tabs.setGeometry(QRect(10, 10, 371, 591))
        self.tabs.setTabShape(QTabWidget.TabShape.Rounded)
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.verticalLayoutWidget = QWidget(self.tab)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QRect(10, 10, 351, 541))
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setSpacing(-1)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.pushButton = QPushButton(self.verticalLayoutWidget)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setCheckable(False)
        self.pushButton.setAutoDefault(False)

        self.horizontalLayout_2.addWidget(self.pushButton, 0, Qt.AlignmentFlag.AlignTop)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.tabs.addTab(self.tab, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.tabs.addTab(self.tab_2, "")
        self.tab_3 = QWidget()
        self.tab_3.setObjectName(u"tab_3")
        self.tabs.addTab(self.tab_3, "")
        self.tab_4 = QWidget()
        self.tab_4.setObjectName(u"tab_4")
        self.tabs.addTab(self.tab_4, "")

        self.retranslateUi(octron_gui)

        self.tabs.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(octron_gui)

    def retranslateUi(self, octron_gui):
        octron_gui.setWindowTitle(QCoreApplication.translate("octron_gui", u"octron_gui", None))
        self.pushButton.setText(QCoreApplication.translate("octron_gui", u"PushButton", None))
        self.tabs.setTabText(self.tabs.indexOf(self.tab), QCoreApplication.translate("octron_gui", u"Project", None))
        self.tabs.setTabText(self.tabs.indexOf(self.tab_2), QCoreApplication.translate("octron_gui", u"Segment", None))
        self.tabs.setTabText(self.tabs.indexOf(self.tab_3), QCoreApplication.translate("octron_gui", u"Train", None))
        self.tabs.setTabText(self.tabs.indexOf(self.tab_4), QCoreApplication.translate("octron_gui", u"Predict", None))

