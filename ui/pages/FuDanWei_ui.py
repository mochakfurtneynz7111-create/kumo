# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'FuDanWei.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QScrollArea, QSizePolicy,
    QSpacerItem, QTextEdit, QVBoxLayout, QWidget)

class Ui_FuDanWei(object):
    def setupUi(self, FuDanWei):
        if not FuDanWei.objectName():
            FuDanWei.setObjectName(u"FuDanWei")
        FuDanWei.resize(742, 582)
        self.verticalLayout_2 = QVBoxLayout(FuDanWei)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.scrollArea = QScrollArea(FuDanWei)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setStyleSheet(u"")
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents_2 = QWidget()
        self.scrollAreaWidgetContents_2.setObjectName(u"scrollAreaWidgetContents_2")
        self.scrollAreaWidgetContents_2.setGeometry(QRect(0, 0, 722, 524))
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollAreaWidgetContents_2.sizePolicy().hasHeightForWidth())
        self.scrollAreaWidgetContents_2.setSizePolicy(sizePolicy)
        self.scrollAreaWidgetContents_2.setMinimumSize(QSize(300, 424))
        self.verticalLayout = QVBoxLayout(self.scrollAreaWidgetContents_2)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label_workspace = QLabel(self.scrollAreaWidgetContents_2)
        self.label_workspace.setObjectName(u"label_workspace")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.label_workspace.sizePolicy().hasHeightForWidth())
        self.label_workspace.setSizePolicy(sizePolicy1)
        self.label_workspace.setMinimumSize(QSize(87, 30))
        self.label_workspace.setMaximumSize(QSize(60, 16777215))
        font = QFont()
        font.setPointSize(8)
        self.label_workspace.setFont(font)

        self.horizontalLayout.addWidget(self.label_workspace)

        self.WorkSpaceLine = QLineEdit(self.scrollAreaWidgetContents_2)
        self.WorkSpaceLine.setObjectName(u"WorkSpaceLine")
        self.WorkSpaceLine.setMinimumSize(QSize(0, 30))
        self.WorkSpaceLine.setMaximumSize(QSize(16777215, 16777215))

        self.horizontalLayout.addWidget(self.WorkSpaceLine)

        self.WorkSpaceButton = QPushButton(self.scrollAreaWidgetContents_2)
        self.WorkSpaceButton.setObjectName(u"WorkSpaceButton")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.WorkSpaceButton.sizePolicy().hasHeightForWidth())
        self.WorkSpaceButton.setSizePolicy(sizePolicy2)
        self.WorkSpaceButton.setMinimumSize(QSize(100, 30))
        self.WorkSpaceButton.setMaximumSize(QSize(100, 16777215))

        self.horizontalLayout.addWidget(self.WorkSpaceButton)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_toml_file = QLabel(self.scrollAreaWidgetContents_2)
        self.label_toml_file.setObjectName(u"label_toml_file")
        sizePolicy.setHeightForWidth(self.label_toml_file.sizePolicy().hasHeightForWidth())
        self.label_toml_file.setSizePolicy(sizePolicy)
        self.label_toml_file.setMinimumSize(QSize(87, 30))
        self.label_toml_file.setFont(font)

        self.horizontalLayout_3.addWidget(self.label_toml_file)

        self.TomlFileComboBox = QComboBox(self.scrollAreaWidgetContents_2)
        self.TomlFileComboBox.setObjectName(u"TomlFileComboBox")
        self.TomlFileComboBox.setMinimumSize(QSize(240, 30))

        self.horizontalLayout_3.addWidget(self.TomlFileComboBox)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_4)


        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label_output = QLabel(self.scrollAreaWidgetContents_2)
        self.label_output.setObjectName(u"label_output")
        self.label_output.setMinimumSize(QSize(87, 30))
        self.label_output.setMaximumSize(QSize(87, 30))
        self.label_output.setFont(font)

        self.horizontalLayout_4.addWidget(self.label_output)

        self.outputEdit = QTextEdit(self.scrollAreaWidgetContents_2)
        self.outputEdit.setObjectName(u"outputEdit")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.outputEdit.sizePolicy().hasHeightForWidth())
        self.outputEdit.setSizePolicy(sizePolicy3)
        self.outputEdit.setMinimumSize(QSize(0, 150))
        self.outputEdit.setReadOnly(True)

        self.horizontalLayout_4.addWidget(self.outputEdit)


        self.verticalLayout.addLayout(self.horizontalLayout_4)

        self.scrollArea.setWidget(self.scrollAreaWidgetContents_2)

        self.verticalLayout_2.addWidget(self.scrollArea)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setSpacing(6)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalSpacer_3 = QSpacerItem(101, 20, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_3)

        self.GenerateButton = QPushButton(FuDanWei)
        self.GenerateButton.setObjectName(u"GenerateButton")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy4.setHorizontalStretch(1)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.GenerateButton.sizePolicy().hasHeightForWidth())
        self.GenerateButton.setSizePolicy(sizePolicy4)
        self.GenerateButton.setMinimumSize(QSize(120, 30))
        self.GenerateButton.setMaximumSize(QSize(100, 30))
        self.GenerateButton.setFont(font)

        self.horizontalLayout_2.addWidget(self.GenerateButton)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer)

        self.pushButton_3 = QPushButton(FuDanWei)
        self.pushButton_3.setObjectName(u"pushButton_3")
        sizePolicy4.setHeightForWidth(self.pushButton_3.sizePolicy().hasHeightForWidth())
        self.pushButton_3.setSizePolicy(sizePolicy4)
        self.pushButton_3.setMinimumSize(QSize(100, 30))
        self.pushButton_3.setMaximumSize(QSize(100, 30))
        self.pushButton_3.setFont(font)

        self.horizontalLayout_2.addWidget(self.pushButton_3)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_2)

        self.pushButton_4 = QPushButton(FuDanWei)
        self.pushButton_4.setObjectName(u"pushButton_4")
        sizePolicy4.setHeightForWidth(self.pushButton_4.sizePolicy().hasHeightForWidth())
        self.pushButton_4.setSizePolicy(sizePolicy4)
        self.pushButton_4.setMinimumSize(QSize(100, 30))
        self.pushButton_4.setMaximumSize(QSize(100, 30))
        self.pushButton_4.setFont(font)

        self.horizontalLayout_2.addWidget(self.pushButton_4)

        self.horizontalSpacer_5 = QSpacerItem(101, 20, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_5)


        self.verticalLayout_2.addLayout(self.horizontalLayout_2)


        self.retranslateUi(FuDanWei)

        QMetaObject.connectSlotsByName(FuDanWei)
    # setupUi

    def retranslateUi(self, FuDanWei):
        FuDanWei.setWindowTitle(QCoreApplication.translate("FuDanWei", u"\u590d\u65e6\u5fae", None))
        self.label_workspace.setText(QCoreApplication.translate("FuDanWei", u"\u5de5\u4f5c\u76ee\u5f55", None))
        self.WorkSpaceButton.setText(QCoreApplication.translate("FuDanWei", u"\u6d4f\u89c8", None))
        self.label_toml_file.setText(QCoreApplication.translate("FuDanWei", u"\u9009\u62e9 Toml \u6587\u4ef6", None))
        self.label_output.setText(QCoreApplication.translate("FuDanWei", u"\u8f93\u51fa", None))
        self.GenerateButton.setText(QCoreApplication.translate("FuDanWei", u"\u751f\u6210 toml \u6587\u4ef6", None))
        self.pushButton_3.setText(QCoreApplication.translate("FuDanWei", u"\u7f16\u8bd1", None))
        self.pushButton_4.setText(QCoreApplication.translate("FuDanWei", u"\u6a21\u62df", None))
    # retranslateUi

