# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ControlPanel.ui'
##
## Created by: Qt User Interface Compiler version 6.6.3
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QSizePolicy, QSplitter, QTabWidget,
    QVBoxLayout, QWidget)

class Ui_ControlPanel(object):
    def setupUi(self, ControlPanel):
        if not ControlPanel.objectName():
            ControlPanel.setObjectName(u"ControlPanel")
        ControlPanel.resize(646, 367)
        self.verticalLayout = QVBoxLayout(ControlPanel)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.splitter = QSplitter(ControlPanel)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setLineWidth(0)
        self.splitter.setOrientation(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(0)
        self.splitter.setChildrenCollapsible(False)
        self.configWidget = QTabWidget(self.splitter)
        self.configWidget.setObjectName(u"configWidget")
        self.configWidget.setTabsClosable(True)
        self.splitter.addWidget(self.configWidget)
        self.dataViewerWidget = QTabWidget(self.splitter)
        self.dataViewerWidget.setObjectName(u"dataViewerWidget")
        self.dataViewerWidget.setTabsClosable(True)
        self.splitter.addWidget(self.dataViewerWidget)

        self.verticalLayout.addWidget(self.splitter)


        self.retranslateUi(ControlPanel)

        self.configWidget.setCurrentIndex(-1)
        self.dataViewerWidget.setCurrentIndex(-1)


        QMetaObject.connectSlotsByName(ControlPanel)
    # setupUi

    def retranslateUi(self, ControlPanel):
        ControlPanel.setWindowTitle(QCoreApplication.translate("ControlPanel", u"Form", None))
    # retranslateUi

