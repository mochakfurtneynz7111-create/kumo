# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_window.ui'
##
## Created by: Qt User Interface Compiler version 6.6.3
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QMainWindow, QSizePolicy, QSplitter,
    QStatusBar, QTabWidget, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1000, 600)
        MainWindow.setStyleSheet(u"")
        self.openFolderAction = QAction(MainWindow)
        self.openFolderAction.setObjectName(u"openFolderAction")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.menuTabWidget = QTabWidget(self.centralwidget)
        self.menuTabWidget.setObjectName(u"menuTabWidget")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.menuTabWidget.sizePolicy().hasHeightForWidth())
        self.menuTabWidget.setSizePolicy(sizePolicy)
        self.menuTabWidget.setMinimumSize(QSize(0, 112))

        self.verticalLayout.addWidget(self.menuTabWidget)

        self.splitter_h = QSplitter(self.centralwidget)
        self.splitter_h.setObjectName(u"splitter_h")
        self.splitter_h.setLineWidth(0)
        self.splitter_h.setOrientation(Qt.Orientation.Horizontal)
        self.splitter_h.setHandleWidth(0)
        self.splitter_left_v = QSplitter(self.splitter_h)
        self.splitter_left_v.setObjectName(u"splitter_left_v")
        self.splitter_left_v.setMinimumSize(QSize(100, 0))
        self.splitter_left_v.setLineWidth(0)
        self.splitter_left_v.setOrientation(Qt.Orientation.Vertical)
        self.splitter_left_v.setHandleWidth(0)
        self.widget_left_up = QWidget(self.splitter_left_v)
        self.widget_left_up.setObjectName(u"widget_left_up")
        self.widget_left_up.setStyleSheet(u"")
        self.verticalLayout_2 = QVBoxLayout(self.widget_left_up)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.splitter_left_v.addWidget(self.widget_left_up)
        self.widget_left_middle = QWidget(self.splitter_left_v)
        self.widget_left_middle.setObjectName(u"widget_left_middle")
        self.widget_left_middle.setStyleSheet(u"")
        self.verticalLayout_3 = QVBoxLayout(self.widget_left_middle)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.splitter_left_v.addWidget(self.widget_left_middle)
        self.widget_left_down = QWidget(self.splitter_left_v)
        self.widget_left_down.setObjectName(u"widget_left_down")
        self.widget_left_down.setStyleSheet(u"")
        self.verticalLayout_4 = QVBoxLayout(self.widget_left_down)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.splitter_left_v.addWidget(self.widget_left_down)
        self.splitter_h.addWidget(self.splitter_left_v)
        self.splitter_right_v = QSplitter(self.splitter_h)
        self.splitter_right_v.setObjectName(u"splitter_right_v")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.splitter_right_v.sizePolicy().hasHeightForWidth())
        self.splitter_right_v.setSizePolicy(sizePolicy1)
        self.splitter_right_v.setMinimumSize(QSize(0, 0))
        self.splitter_right_v.setLineWidth(0)
        self.splitter_right_v.setOrientation(Qt.Orientation.Vertical)
        self.splitter_right_v.setHandleWidth(0)
        self.widget_right_up = QWidget(self.splitter_right_v)
        self.widget_right_up.setObjectName(u"widget_right_up")
        self.widget_right_up.setStyleSheet(u"")
        self.verticalLayout_5 = QVBoxLayout(self.widget_right_up)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.splitter_right_v.addWidget(self.widget_right_up)
        self.widget_right_down = QWidget(self.splitter_right_v)
        self.widget_right_down.setObjectName(u"widget_right_down")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.widget_right_down.sizePolicy().hasHeightForWidth())
        self.widget_right_down.setSizePolicy(sizePolicy2)
        self.widget_right_down.setStyleSheet(u"")
        self.verticalLayout_6 = QVBoxLayout(self.widget_right_down)
        self.verticalLayout_6.setSpacing(6)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.splitter_right_v.addWidget(self.widget_right_down)
        self.splitter_h.addWidget(self.splitter_right_v)

        self.verticalLayout.addWidget(self.splitter_h)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusBar = QStatusBar(MainWindow)
        self.statusBar.setObjectName(u"statusBar")
        MainWindow.setStatusBar(self.statusBar)

        self.retranslateUi(MainWindow)

        self.menuTabWidget.setCurrentIndex(-1)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"\u82af\u7247\u6574\u5408\u5e73\u53f0", None))
        self.openFolderAction.setText(QCoreApplication.translate("MainWindow", u"\u6253\u5f00\u6587\u4ef6\u5939", None))
    # retranslateUi

