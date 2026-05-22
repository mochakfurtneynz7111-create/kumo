# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ConfigDialog.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QLayout, QLineEdit, QPushButton, QScrollArea,
    QSizePolicy, QSpacerItem, QSpinBox, QToolButton,
    QVBoxLayout, QWidget)

class Ui_ConfigDialog(object):
    def setupUi(self, ConfigDialog):
        if not ConfigDialog.objectName():
            ConfigDialog.setObjectName(u"ConfigDialog")
        ConfigDialog.resize(800, 600)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(ConfigDialog.sizePolicy().hasHeightForWidth())
        ConfigDialog.setSizePolicy(sizePolicy)
        ConfigDialog.setMinimumSize(QSize(0, 0))
        self.verticalLayout_3 = QVBoxLayout(ConfigDialog)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.scrollArea = QScrollArea(ConfigDialog)
        self.scrollArea.setObjectName(u"scrollArea")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.scrollArea.sizePolicy().hasHeightForWidth())
        self.scrollArea.setSizePolicy(sizePolicy1)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 768, 1855))
        sizePolicy.setHeightForWidth(self.scrollAreaWidgetContents.sizePolicy().hasHeightForWidth())
        self.scrollAreaWidgetContents.setSizePolicy(sizePolicy)
        self.verticalLayout = QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.ParseGroupBox = QGroupBox(self.scrollAreaWidgetContents)
        self.ParseGroupBox.setObjectName(u"ParseGroupBox")
        sizePolicy.setHeightForWidth(self.ParseGroupBox.sizePolicy().hasHeightForWidth())
        self.ParseGroupBox.setSizePolicy(sizePolicy)
        self.ParseGroupBox.setMinimumSize(QSize(0, 0))
        self.ParseGroupBox.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
        self.verticalLayout_2 = QVBoxLayout(self.ParseGroupBox)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.label_input = QLabel(self.ParseGroupBox)
        self.label_input.setObjectName(u"label_input")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.label_input.sizePolicy().hasHeightForWidth())
        self.label_input.setSizePolicy(sizePolicy2)
        self.label_input.setMinimumSize(QSize(62, 0))

        self.horizontalLayout_7.addWidget(self.label_input)

        self.ShapeInput = QLineEdit(self.ParseGroupBox)
        self.ShapeInput.setObjectName(u"ShapeInput")

        self.horizontalLayout_7.addWidget(self.ShapeInput)


        self.gridLayout.addLayout(self.horizontalLayout_7, 4, 0, 1, 2)

        self.horizontalLayout_21 = QHBoxLayout()
        self.horizontalLayout_21.setObjectName(u"horizontalLayout_21")
        self.label_weights = QLabel(self.ParseGroupBox)
        self.label_weights.setObjectName(u"label_weights")
        sizePolicy2.setHeightForWidth(self.label_weights.sizePolicy().hasHeightForWidth())
        self.label_weights.setSizePolicy(sizePolicy2)
        self.label_weights.setMinimumSize(QSize(62, 0))

        self.horizontalLayout_21.addWidget(self.label_weights)

        self.WeightInput = QLineEdit(self.ParseGroupBox)
        self.WeightInput.setObjectName(u"WeightInput")

        self.horizontalLayout_21.addWidget(self.WeightInput)


        self.gridLayout.addLayout(self.horizontalLayout_21, 6, 0, 1, 2)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_framework = QLabel(self.ParseGroupBox)
        self.label_framework.setObjectName(u"label_framework")
        sizePolicy2.setHeightForWidth(self.label_framework.sizePolicy().hasHeightForWidth())
        self.label_framework.setSizePolicy(sizePolicy2)
        self.label_framework.setMinimumSize(QSize(0, 0))

        self.horizontalLayout_3.addWidget(self.label_framework)

        self.FrameWorkComboBox = QComboBox(self.ParseGroupBox)
        self.FrameWorkComboBox.setObjectName(u"FrameWorkComboBox")
        sizePolicy2.setHeightForWidth(self.FrameWorkComboBox.sizePolicy().hasHeightForWidth())
        self.FrameWorkComboBox.setSizePolicy(sizePolicy2)

        self.horizontalLayout_3.addWidget(self.FrameWorkComboBox)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_3)


        self.gridLayout.addLayout(self.horizontalLayout_3, 3, 0, 1, 1)

        self.horizontalLayout_22 = QHBoxLayout()
        self.horizontalLayout_22.setObjectName(u"horizontalLayout_22")
        self.label_custom_op = QLabel(self.ParseGroupBox)
        self.label_custom_op.setObjectName(u"label_custom_op")
        sizePolicy2.setHeightForWidth(self.label_custom_op.sizePolicy().hasHeightForWidth())
        self.label_custom_op.setSizePolicy(sizePolicy2)
        self.label_custom_op.setMinimumSize(QSize(76, 0))
        self.label_custom_op.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.horizontalLayout_22.addWidget(self.label_custom_op)

        self.CustomOpInput = QLineEdit(self.ParseGroupBox)
        self.CustomOpInput.setObjectName(u"CustomOpInput")

        self.horizontalLayout_22.addWidget(self.CustomOpInput)


        self.gridLayout.addLayout(self.horizontalLayout_22, 6, 2, 1, 2)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.label_pre_method = QLabel(self.ParseGroupBox)
        self.label_pre_method.setObjectName(u"label_pre_method")
        sizePolicy2.setHeightForWidth(self.label_pre_method.sizePolicy().hasHeightForWidth())
        self.label_pre_method.setSizePolicy(sizePolicy2)
        self.label_pre_method.setMinimumSize(QSize(76, 0))
        self.label_pre_method.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_6.addWidget(self.label_pre_method)

        self.PreMethodComboBox = QComboBox(self.ParseGroupBox)
        self.PreMethodComboBox.setObjectName(u"PreMethodComboBox")

        self.horizontalLayout_6.addWidget(self.PreMethodComboBox)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_6.addItem(self.horizontalSpacer_2)


        self.gridLayout.addLayout(self.horizontalLayout_6, 3, 3, 1, 1)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_name = QLabel(self.ParseGroupBox)
        self.label_name.setObjectName(u"label_name")
        sizePolicy2.setHeightForWidth(self.label_name.sizePolicy().hasHeightForWidth())
        self.label_name.setSizePolicy(sizePolicy2)
        self.label_name.setMinimumSize(QSize(62, 0))

        self.horizontalLayout_2.addWidget(self.label_name)

        self.NameInput = QLineEdit(self.ParseGroupBox)
        self.NameInput.setObjectName(u"NameInput")

        self.horizontalLayout_2.addWidget(self.NameInput)


        self.gridLayout.addLayout(self.horizontalLayout_2, 2, 0, 1, 4)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label_network = QLabel(self.ParseGroupBox)
        self.label_network.setObjectName(u"label_network")
        sizePolicy2.setHeightForWidth(self.label_network.sizePolicy().hasHeightForWidth())
        self.label_network.setSizePolicy(sizePolicy2)
        self.label_network.setMinimumSize(QSize(62, 0))

        self.horizontalLayout.addWidget(self.label_network)

        self.NetworkInput = QLineEdit(self.ParseGroupBox)
        self.NetworkInput.setObjectName(u"NetworkInput")

        self.horizontalLayout.addWidget(self.NetworkInput)

        self.ChooseNetworkButton = QPushButton(self.ParseGroupBox)
        self.ChooseNetworkButton.setObjectName(u"ChooseNetworkButton")
        sizePolicy2.setHeightForWidth(self.ChooseNetworkButton.sizePolicy().hasHeightForWidth())
        self.ChooseNetworkButton.setSizePolicy(sizePolicy2)
        self.ChooseNetworkButton.setMinimumSize(QSize(0, 0))
        self.ChooseNetworkButton.setMaximumSize(QSize(16777215, 16777215))

        self.horizontalLayout.addWidget(self.ChooseNetworkButton)


        self.gridLayout.addLayout(self.horizontalLayout, 1, 0, 1, 4)

        self.horizontalLayout_20 = QHBoxLayout()
        self.horizontalLayout_20.setObjectName(u"horizontalLayout_20")
        self.label_pre_scale = QLabel(self.ParseGroupBox)
        self.label_pre_scale.setObjectName(u"label_pre_scale")
        sizePolicy2.setHeightForWidth(self.label_pre_scale.sizePolicy().hasHeightForWidth())
        self.label_pre_scale.setSizePolicy(sizePolicy2)
        self.label_pre_scale.setMinimumSize(QSize(76, 0))
        self.label_pre_scale.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.horizontalLayout_20.addWidget(self.label_pre_scale)

        self.PreScaleInput = QLineEdit(self.ParseGroupBox)
        self.PreScaleInput.setObjectName(u"PreScaleInput")

        self.horizontalLayout_20.addWidget(self.PreScaleInput)


        self.gridLayout.addLayout(self.horizontalLayout_20, 5, 2, 1, 2)

        self.horizontalLayout_19 = QHBoxLayout()
        self.horizontalLayout_19.setObjectName(u"horizontalLayout_19")
        self.label_pre_mean = QLabel(self.ParseGroupBox)
        self.label_pre_mean.setObjectName(u"label_pre_mean")
        sizePolicy2.setHeightForWidth(self.label_pre_mean.sizePolicy().hasHeightForWidth())
        self.label_pre_mean.setSizePolicy(sizePolicy2)
        self.label_pre_mean.setMinimumSize(QSize(62, 0))

        self.horizontalLayout_19.addWidget(self.label_pre_mean)

        self.PreMeanInput = QLineEdit(self.ParseGroupBox)
        self.PreMeanInput.setObjectName(u"PreMeanInput")

        self.horizontalLayout_19.addWidget(self.PreMeanInput)


        self.gridLayout.addLayout(self.horizontalLayout_19, 5, 0, 1, 2)

        self.horizontalLayout_18 = QHBoxLayout()
        self.horizontalLayout_18.setObjectName(u"horizontalLayout_18")
        self.label_chann_swap = QLabel(self.ParseGroupBox)
        self.label_chann_swap.setObjectName(u"label_chann_swap")
        sizePolicy2.setHeightForWidth(self.label_chann_swap.sizePolicy().hasHeightForWidth())
        self.label_chann_swap.setSizePolicy(sizePolicy2)
        self.label_chann_swap.setMinimumSize(QSize(76, 0))
        self.label_chann_swap.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_18.addWidget(self.label_chann_swap)

        self.ChannelSwapInput = QLineEdit(self.ParseGroupBox)
        self.ChannelSwapInput.setObjectName(u"ChannelSwapInput")

        self.horizontalLayout_18.addWidget(self.ChannelSwapInput)


        self.gridLayout.addLayout(self.horizontalLayout_18, 4, 3, 1, 1)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.label_input_format = QLabel(self.ParseGroupBox)
        self.label_input_format.setObjectName(u"label_input_format")
        sizePolicy2.setHeightForWidth(self.label_input_format.sizePolicy().hasHeightForWidth())
        self.label_input_format.setSizePolicy(sizePolicy2)
        self.label_input_format.setMinimumSize(QSize(76, 0))
        self.label_input_format.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_8.addWidget(self.label_input_format)

        self.InputFormatComboBox = QComboBox(self.ParseGroupBox)
        self.InputFormatComboBox.setObjectName(u"InputFormatComboBox")

        self.horizontalLayout_8.addWidget(self.InputFormatComboBox)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_8.addItem(self.horizontalSpacer_5)


        self.gridLayout.addLayout(self.horizontalLayout_8, 4, 2, 1, 1)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label_frame_version = QLabel(self.ParseGroupBox)
        self.label_frame_version.setObjectName(u"label_frame_version")
        sizePolicy2.setHeightForWidth(self.label_frame_version.sizePolicy().hasHeightForWidth())
        self.label_frame_version.setSizePolicy(sizePolicy2)
        self.label_frame_version.setMinimumSize(QSize(0, 0))
        self.label_frame_version.setTextFormat(Qt.TextFormat.AutoText)
        self.label_frame_version.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_4.addWidget(self.label_frame_version)

        self.FrameVersionComboBox = QComboBox(self.ParseGroupBox)
        self.FrameVersionComboBox.setObjectName(u"FrameVersionComboBox")
        sizePolicy2.setHeightForWidth(self.FrameVersionComboBox.sizePolicy().hasHeightForWidth())
        self.FrameVersionComboBox.setSizePolicy(sizePolicy2)

        self.horizontalLayout_4.addWidget(self.FrameVersionComboBox)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer)


        self.gridLayout.addLayout(self.horizontalLayout_4, 3, 1, 1, 1)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label_post_method = QLabel(self.ParseGroupBox)
        self.label_post_method.setObjectName(u"label_post_method")
        sizePolicy2.setHeightForWidth(self.label_post_method.sizePolicy().hasHeightForWidth())
        self.label_post_method.setSizePolicy(sizePolicy2)
        self.label_post_method.setMinimumSize(QSize(76, 0))
        self.label_post_method.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_5.addWidget(self.label_post_method)

        self.PostMethodComboBox = QComboBox(self.ParseGroupBox)
        self.PostMethodComboBox.setObjectName(u"PostMethodComboBox")

        self.horizontalLayout_5.addWidget(self.PostMethodComboBox)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_4)


        self.gridLayout.addLayout(self.horizontalLayout_5, 3, 2, 1, 1)

        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 1)
        self.gridLayout.setColumnStretch(2, 1)
        self.gridLayout.setColumnStretch(3, 1)

        self.verticalLayout_2.addLayout(self.gridLayout)


        self.verticalLayout.addWidget(self.ParseGroupBox)

        self.OptimizeGroupBox = QGroupBox(self.scrollAreaWidgetContents)
        self.OptimizeGroupBox.setObjectName(u"OptimizeGroupBox")
        sizePolicy.setHeightForWidth(self.OptimizeGroupBox.sizePolicy().hasHeightForWidth())
        self.OptimizeGroupBox.setSizePolicy(sizePolicy)
        self.OptimizeGroupBox.setMinimumSize(QSize(0, 0))
        self.OptimizeGroupBox.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
        self.verticalLayout_5 = QVBoxLayout(self.OptimizeGroupBox)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.horizontalLayout_11 = QHBoxLayout()
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.label_debug = QLabel(self.OptimizeGroupBox)
        self.label_debug.setObjectName(u"label_debug")
        sizePolicy2.setHeightForWidth(self.label_debug.sizePolicy().hasHeightForWidth())
        self.label_debug.setSizePolicy(sizePolicy2)
        self.label_debug.setMinimumSize(QSize(0, 0))

        self.horizontalLayout_9.addWidget(self.label_debug)

        self.DebugCheckBox = QCheckBox(self.OptimizeGroupBox)
        self.DebugCheckBox.setObjectName(u"DebugCheckBox")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.DebugCheckBox.sizePolicy().hasHeightForWidth())
        self.DebugCheckBox.setSizePolicy(sizePolicy3)

        self.horizontalLayout_9.addWidget(self.DebugCheckBox)

        self.horizontalSpacer_34 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_9.addItem(self.horizontalSpacer_34)


        self.horizontalLayout_11.addLayout(self.horizontalLayout_9)

        self.horizontalLayout_10 = QHBoxLayout()
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.label_target = QLabel(self.OptimizeGroupBox)
        self.label_target.setObjectName(u"label_target")
        sizePolicy2.setHeightForWidth(self.label_target.sizePolicy().hasHeightForWidth())
        self.label_target.setSizePolicy(sizePolicy2)
        self.label_target.setMinimumSize(QSize(0, 0))

        self.horizontalLayout_10.addWidget(self.label_target)

        self.TargetComboBox = QComboBox(self.OptimizeGroupBox)
        self.TargetComboBox.setObjectName(u"TargetComboBox")

        self.horizontalLayout_10.addWidget(self.TargetComboBox)

        self.horizontalSpacer_35 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_10.addItem(self.horizontalSpacer_35)


        self.horizontalLayout_11.addLayout(self.horizontalLayout_10)

        self.horizontalSpacer_6 = QSpacerItem(40, 26, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_11.addItem(self.horizontalSpacer_6)

        self.horizontalLayout_11.setStretch(0, 1)
        self.horizontalLayout_11.setStretch(1, 1)
        self.horizontalLayout_11.setStretch(2, 2)

        self.verticalLayout_5.addLayout(self.horizontalLayout_11)

        self.OptimizeOptionToolButton = QToolButton(self.OptimizeGroupBox)
        self.OptimizeOptionToolButton.setObjectName(u"OptimizeOptionToolButton")
        sizePolicy2.setHeightForWidth(self.OptimizeOptionToolButton.sizePolicy().hasHeightForWidth())
        self.OptimizeOptionToolButton.setSizePolicy(sizePolicy2)
        self.OptimizeOptionToolButton.setCursor(QCursor(Qt.ArrowCursor))
        self.OptimizeOptionToolButton.setMouseTracking(False)
        self.OptimizeOptionToolButton.setTabletTracking(False)
        self.OptimizeOptionToolButton.setStyleSheet(u" QToolButton {\n"
"                text-align: center;  /* \u4f7f\u6587\u672c\u5c45\u4e2d */\n"
"                padding-right: 16px;  /* \u786e\u4fdd\u7bad\u5934\u548c\u6587\u672c\u6709\u9002\u5f53\u7684\u95f4\u8ddd */\n"
"            }")
        self.OptimizeOptionToolButton.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.OptimizeOptionToolButton.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.OptimizeOptionToolButton.setAutoRaise(False)
        self.OptimizeOptionToolButton.setArrowType(Qt.ArrowType.RightArrow)

        self.verticalLayout_5.addWidget(self.OptimizeOptionToolButton)

        self.OptimizeOptionGroupBox = QGroupBox(self.OptimizeGroupBox)
        self.OptimizeOptionGroupBox.setObjectName(u"OptimizeOptionGroupBox")
        sizePolicy2.setHeightForWidth(self.OptimizeOptionGroupBox.sizePolicy().hasHeightForWidth())
        self.OptimizeOptionGroupBox.setSizePolicy(sizePolicy2)
        self.OptimizeOptionGroupBox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.OptimizeOptionGroupBox.setFlat(False)
        self.OptimizeOptionGroupBox.setCheckable(False)
        self.verticalLayout_6 = QVBoxLayout(self.OptimizeOptionGroupBox)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.horizontalLayout_12 = QHBoxLayout()
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.label_customop_config = QLabel(self.OptimizeOptionGroupBox)
        self.label_customop_config.setObjectName(u"label_customop_config")
        sizePolicy2.setHeightForWidth(self.label_customop_config.sizePolicy().hasHeightForWidth())
        self.label_customop_config.setSizePolicy(sizePolicy2)
        self.label_customop_config.setMinimumSize(QSize(117, 0))

        self.horizontalLayout_12.addWidget(self.label_customop_config)

        self.CustomopConfigOpInput = QLineEdit(self.OptimizeOptionGroupBox)
        self.CustomopConfigOpInput.setObjectName(u"CustomopConfigOpInput")

        self.horizontalLayout_12.addWidget(self.CustomopConfigOpInput)


        self.verticalLayout_4.addLayout(self.horizontalLayout_12)

        self.horizontalLayout_13 = QHBoxLayout()
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.label_config = QLabel(self.OptimizeOptionGroupBox)
        self.label_config.setObjectName(u"label_config")
        sizePolicy2.setHeightForWidth(self.label_config.sizePolicy().hasHeightForWidth())
        self.label_config.setSizePolicy(sizePolicy2)
        self.label_config.setMinimumSize(QSize(117, 0))

        self.horizontalLayout_13.addWidget(self.label_config)

        self.ConfigInput = QLineEdit(self.OptimizeOptionGroupBox)
        self.ConfigInput.setObjectName(u"ConfigInput")

        self.horizontalLayout_13.addWidget(self.ConfigInput)


        self.verticalLayout_4.addLayout(self.horizontalLayout_13)

        self.horizontalLayout_17 = QHBoxLayout()
        self.horizontalLayout_17.setObjectName(u"horizontalLayout_17")
        self.horizontalLayout_14 = QHBoxLayout()
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.label_customop_on = QLabel(self.OptimizeOptionGroupBox)
        self.label_customop_on.setObjectName(u"label_customop_on")
        sizePolicy.setHeightForWidth(self.label_customop_on.sizePolicy().hasHeightForWidth())
        self.label_customop_on.setSizePolicy(sizePolicy)
        self.label_customop_on.setMinimumSize(QSize(117, 0))

        self.horizontalLayout_14.addWidget(self.label_customop_on)

        self.CustomOpOnInput = QLineEdit(self.OptimizeOptionGroupBox)
        self.CustomOpOnInput.setObjectName(u"CustomOpOnInput")

        self.horizontalLayout_14.addWidget(self.CustomOpOnInput)


        self.horizontalLayout_17.addLayout(self.horizontalLayout_14)

        self.horizontalLayout_15 = QHBoxLayout()
        self.horizontalLayout_15.setObjectName(u"horizontalLayout_15")
        self.label_pass_on = QLabel(self.OptimizeOptionGroupBox)
        self.label_pass_on.setObjectName(u"label_pass_on")
        sizePolicy.setHeightForWidth(self.label_pass_on.sizePolicy().hasHeightForWidth())
        self.label_pass_on.setSizePolicy(sizePolicy)
        self.label_pass_on.setMinimumSize(QSize(75, 0))

        self.horizontalLayout_15.addWidget(self.label_pass_on)

        self.PassOnInput = QLineEdit(self.OptimizeOptionGroupBox)
        self.PassOnInput.setObjectName(u"PassOnInput")

        self.horizontalLayout_15.addWidget(self.PassOnInput)


        self.horizontalLayout_17.addLayout(self.horizontalLayout_15)

        self.horizontalLayout_16 = QHBoxLayout()
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.label_pass_off = QLabel(self.OptimizeOptionGroupBox)
        self.label_pass_off.setObjectName(u"label_pass_off")
        sizePolicy.setHeightForWidth(self.label_pass_off.sizePolicy().hasHeightForWidth())
        self.label_pass_off.setSizePolicy(sizePolicy)
        self.label_pass_off.setMinimumSize(QSize(75, 0))

        self.horizontalLayout_16.addWidget(self.label_pass_off)

        self.PassOffInput = QLineEdit(self.OptimizeOptionGroupBox)
        self.PassOffInput.setObjectName(u"PassOffInput")

        self.horizontalLayout_16.addWidget(self.PassOffInput)


        self.horizontalLayout_17.addLayout(self.horizontalLayout_16)


        self.verticalLayout_4.addLayout(self.horizontalLayout_17)


        self.verticalLayout_6.addLayout(self.verticalLayout_4)


        self.verticalLayout_5.addWidget(self.OptimizeOptionGroupBox)


        self.verticalLayout.addWidget(self.OptimizeGroupBox)

        self.QuantizeGoupBox = QGroupBox(self.scrollAreaWidgetContents)
        self.QuantizeGoupBox.setObjectName(u"QuantizeGoupBox")
        self.QuantizeGoupBox.setMinimumSize(QSize(0, 0))
        self.QuantizeGoupBox.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
        self.verticalLayout_8 = QVBoxLayout(self.QuantizeGoupBox)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setHorizontalSpacing(6)
        self.gridLayout_2.setContentsMargins(-1, -1, 0, -1)
        self.horizontalLayout_25 = QHBoxLayout()
        self.horizontalLayout_25.setObjectName(u"horizontalLayout_25")
        self.label_forward_list = QLabel(self.QuantizeGoupBox)
        self.label_forward_list.setObjectName(u"label_forward_list")
        sizePolicy.setHeightForWidth(self.label_forward_list.sizePolicy().hasHeightForWidth())
        self.label_forward_list.setSizePolicy(sizePolicy)
        self.label_forward_list.setMinimumSize(QSize(82, 0))

        self.horizontalLayout_25.addWidget(self.label_forward_list)

        self.ForwardListLineEdit = QLineEdit(self.QuantizeGoupBox)
        self.ForwardListLineEdit.setObjectName(u"ForwardListLineEdit")

        self.horizontalLayout_25.addWidget(self.ForwardListLineEdit)


        self.gridLayout_2.addLayout(self.horizontalLayout_25, 2, 0, 1, 4)

        self.horizontalLayout_28 = QHBoxLayout()
        self.horizontalLayout_28.setObjectName(u"horizontalLayout_28")
        self.label_bits = QLabel(self.QuantizeGoupBox)
        self.label_bits.setObjectName(u"label_bits")
        sizePolicy.setHeightForWidth(self.label_bits.sizePolicy().hasHeightForWidth())
        self.label_bits.setSizePolicy(sizePolicy)
        self.label_bits.setMinimumSize(QSize(82, 0))

        self.horizontalLayout_28.addWidget(self.label_bits)

        self.BitsComboBox = QComboBox(self.QuantizeGoupBox)
        self.BitsComboBox.setObjectName(u"BitsComboBox")

        self.horizontalLayout_28.addWidget(self.BitsComboBox)

        self.horizontalSpacer_10 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_28.addItem(self.horizontalSpacer_10)


        self.gridLayout_2.addLayout(self.horizontalLayout_28, 0, 2, 1, 1)

        self.horizontalLayout_27 = QHBoxLayout()
        self.horizontalLayout_27.setObjectName(u"horizontalLayout_27")
        self.label_per = QLabel(self.QuantizeGoupBox)
        self.label_per.setObjectName(u"label_per")
        sizePolicy.setHeightForWidth(self.label_per.sizePolicy().hasHeightForWidth())
        self.label_per.setSizePolicy(sizePolicy)
        self.label_per.setMinimumSize(QSize(82, 0))

        self.horizontalLayout_27.addWidget(self.label_per)

        self.PerComboBox = QComboBox(self.QuantizeGoupBox)
        self.PerComboBox.setObjectName(u"PerComboBox")
        sizePolicy2.setHeightForWidth(self.PerComboBox.sizePolicy().hasHeightForWidth())
        self.PerComboBox.setSizePolicy(sizePolicy2)

        self.horizontalLayout_27.addWidget(self.PerComboBox)

        self.horizontalSpacer_9 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_27.addItem(self.horizontalSpacer_9)


        self.gridLayout_2.addLayout(self.horizontalLayout_27, 0, 1, 1, 1)

        self.horizontalLayout_24 = QHBoxLayout()
        self.horizontalLayout_24.setObjectName(u"horizontalLayout_24")
        self.label_forward_dir = QLabel(self.QuantizeGoupBox)
        self.label_forward_dir.setObjectName(u"label_forward_dir")
        sizePolicy.setHeightForWidth(self.label_forward_dir.sizePolicy().hasHeightForWidth())
        self.label_forward_dir.setSizePolicy(sizePolicy)
        self.label_forward_dir.setMinimumSize(QSize(82, 0))
        self.label_forward_dir.setMaximumSize(QSize(16777215, 16777215))

        self.horizontalLayout_24.addWidget(self.label_forward_dir)

        self.ForwardDirLineEdit = QLineEdit(self.QuantizeGoupBox)
        self.ForwardDirLineEdit.setObjectName(u"ForwardDirLineEdit")
        sizePolicy3.setHeightForWidth(self.ForwardDirLineEdit.sizePolicy().hasHeightForWidth())
        self.ForwardDirLineEdit.setSizePolicy(sizePolicy3)

        self.horizontalLayout_24.addWidget(self.ForwardDirLineEdit)


        self.gridLayout_2.addLayout(self.horizontalLayout_24, 1, 0, 1, 4)

        self.horizontalLayout_29 = QHBoxLayout()
        self.horizontalLayout_29.setObjectName(u"horizontalLayout_29")
        self.label_saturation = QLabel(self.QuantizeGoupBox)
        self.label_saturation.setObjectName(u"label_saturation")
        sizePolicy.setHeightForWidth(self.label_saturation.sizePolicy().hasHeightForWidth())
        self.label_saturation.setSizePolicy(sizePolicy)
        self.label_saturation.setMinimumSize(QSize(82, 0))

        self.horizontalLayout_29.addWidget(self.label_saturation)

        self.SaturationComboBox = QComboBox(self.QuantizeGoupBox)
        self.SaturationComboBox.setObjectName(u"SaturationComboBox")

        self.horizontalLayout_29.addWidget(self.SaturationComboBox)

        self.horizontalSpacer_11 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_29.addItem(self.horizontalSpacer_11)


        self.gridLayout_2.addLayout(self.horizontalLayout_29, 0, 3, 1, 1)

        self.horizontalLayout_26 = QHBoxLayout()
        self.horizontalLayout_26.setObjectName(u"horizontalLayout_26")
        self.label_forward_mode = QLabel(self.QuantizeGoupBox)
        self.label_forward_mode.setObjectName(u"label_forward_mode")
        sizePolicy.setHeightForWidth(self.label_forward_mode.sizePolicy().hasHeightForWidth())
        self.label_forward_mode.setSizePolicy(sizePolicy)
        self.label_forward_mode.setMinimumSize(QSize(82, 0))
        self.label_forward_mode.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_26.addWidget(self.label_forward_mode)

        self.ForwardModeComboBox = QComboBox(self.QuantizeGoupBox)
        self.ForwardModeComboBox.setObjectName(u"ForwardModeComboBox")
        sizePolicy2.setHeightForWidth(self.ForwardModeComboBox.sizePolicy().hasHeightForWidth())
        self.ForwardModeComboBox.setSizePolicy(sizePolicy2)

        self.horizontalLayout_26.addWidget(self.ForwardModeComboBox)

        self.horizontalSpacer_8 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_26.addItem(self.horizontalSpacer_8)


        self.gridLayout_2.addLayout(self.horizontalLayout_26, 0, 0, 1, 1)

        self.gridLayout_2.setColumnStretch(0, 1)
        self.gridLayout_2.setColumnStretch(1, 1)
        self.gridLayout_2.setColumnStretch(2, 1)
        self.gridLayout_2.setColumnStretch(3, 1)

        self.verticalLayout_8.addLayout(self.gridLayout_2)

        self.QuantizeOptionToolButton = QToolButton(self.QuantizeGoupBox)
        self.QuantizeOptionToolButton.setObjectName(u"QuantizeOptionToolButton")
        sizePolicy2.setHeightForWidth(self.QuantizeOptionToolButton.sizePolicy().hasHeightForWidth())
        self.QuantizeOptionToolButton.setSizePolicy(sizePolicy2)
        self.QuantizeOptionToolButton.setCursor(QCursor(Qt.ArrowCursor))
        self.QuantizeOptionToolButton.setMouseTracking(False)
        self.QuantizeOptionToolButton.setTabletTracking(False)
        self.QuantizeOptionToolButton.setStyleSheet(u" QToolButton {\n"
"                text-align: center;  /* \u4f7f\u6587\u672c\u5c45\u4e2d */\n"
"                padding-right: 16px;  /* \u786e\u4fdd\u7bad\u5934\u548c\u6587\u672c\u6709\u9002\u5f53\u7684\u95f4\u8ddd */\n"
"            }")
        self.QuantizeOptionToolButton.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.QuantizeOptionToolButton.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.QuantizeOptionToolButton.setAutoRaise(False)
        self.QuantizeOptionToolButton.setArrowType(Qt.ArrowType.RightArrow)

        self.verticalLayout_8.addWidget(self.QuantizeOptionToolButton)

        self.QuantizeOptionGroupBox = QGroupBox(self.QuantizeGoupBox)
        self.QuantizeOptionGroupBox.setObjectName(u"QuantizeOptionGroupBox")
        sizePolicy2.setHeightForWidth(self.QuantizeOptionGroupBox.sizePolicy().hasHeightForWidth())
        self.QuantizeOptionGroupBox.setSizePolicy(sizePolicy2)
        self.QuantizeOptionGroupBox.setMinimumSize(QSize(0, 0))
        self.QuantizeOptionGroupBox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.QuantizeOptionGroupBox.setFlat(False)
        self.QuantizeOptionGroupBox.setCheckable(False)
        self.verticalLayout_7 = QVBoxLayout(self.QuantizeOptionGroupBox)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(-1, 9, -1, -1)
        self.gridLayout_3 = QGridLayout()
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.horizontalLayout_30 = QHBoxLayout()
        self.horizontalLayout_30.setObjectName(u"horizontalLayout_30")
        self.label_batch = QLabel(self.QuantizeOptionGroupBox)
        self.label_batch.setObjectName(u"label_batch")
        self.label_batch.setMinimumSize(QSize(80, 0))

        self.horizontalLayout_30.addWidget(self.label_batch)

        self.BatchLineEdit = QLineEdit(self.QuantizeOptionGroupBox)
        self.BatchLineEdit.setObjectName(u"BatchLineEdit")

        self.horizontalLayout_30.addWidget(self.BatchLineEdit)


        self.gridLayout_3.addLayout(self.horizontalLayout_30, 0, 0, 1, 1)

        self.horizontalLayout_34 = QHBoxLayout()
        self.horizontalLayout_34.setObjectName(u"horizontalLayout_34")
        self.label_no_transinput = QLabel(self.QuantizeOptionGroupBox)
        self.label_no_transinput.setObjectName(u"label_no_transinput")
        self.label_no_transinput.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_34.addWidget(self.label_no_transinput)

        self.NoTransinputcheckBox = QCheckBox(self.QuantizeOptionGroupBox)
        self.NoTransinputcheckBox.setObjectName(u"NoTransinputcheckBox")
        sizePolicy3.setHeightForWidth(self.NoTransinputcheckBox.sizePolicy().hasHeightForWidth())
        self.NoTransinputcheckBox.setSizePolicy(sizePolicy3)

        self.horizontalLayout_34.addWidget(self.NoTransinputcheckBox)

        self.horizontalSpacer_37 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_34.addItem(self.horizontalSpacer_37)


        self.gridLayout_3.addLayout(self.horizontalLayout_34, 1, 1, 1, 1)

        self.horizontalLayout_31 = QHBoxLayout()
        self.horizontalLayout_31.setObjectName(u"horizontalLayout_31")
        self.label_bin_num = QLabel(self.QuantizeOptionGroupBox)
        self.label_bin_num.setObjectName(u"label_bin_num")
        self.label_bin_num.setMinimumSize(QSize(75, 0))

        self.horizontalLayout_31.addWidget(self.label_bin_num)

        self.BinNumLineEdit = QLineEdit(self.QuantizeOptionGroupBox)
        self.BinNumLineEdit.setObjectName(u"BinNumLineEdit")

        self.horizontalLayout_31.addWidget(self.BinNumLineEdit)


        self.gridLayout_3.addLayout(self.horizontalLayout_31, 0, 1, 1, 1)

        self.horizontalLayout_35 = QHBoxLayout()
        self.horizontalLayout_35.setObjectName(u"horizontalLayout_35")
        self.label_before_relu = QLabel(self.QuantizeOptionGroupBox)
        self.label_before_relu.setObjectName(u"label_before_relu")
        self.label_before_relu.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_35.addWidget(self.label_before_relu)

        self.BeforeReluCheckBox = QCheckBox(self.QuantizeOptionGroupBox)
        self.BeforeReluCheckBox.setObjectName(u"BeforeReluCheckBox")
        sizePolicy3.setHeightForWidth(self.BeforeReluCheckBox.sizePolicy().hasHeightForWidth())
        self.BeforeReluCheckBox.setSizePolicy(sizePolicy3)
        self.BeforeReluCheckBox.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.BeforeReluCheckBox.setChecked(False)
        self.BeforeReluCheckBox.setTristate(False)

        self.horizontalLayout_35.addWidget(self.BeforeReluCheckBox)

        self.horizontalSpacer_38 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_35.addItem(self.horizontalSpacer_38)


        self.gridLayout_3.addLayout(self.horizontalLayout_35, 1, 2, 1, 1)

        self.horizontalLayout_33 = QHBoxLayout()
        self.horizontalLayout_33.setObjectName(u"horizontalLayout_33")
        self.label_only_norm = QLabel(self.QuantizeOptionGroupBox)
        self.label_only_norm.setObjectName(u"label_only_norm")
        sizePolicy.setHeightForWidth(self.label_only_norm.sizePolicy().hasHeightForWidth())
        self.label_only_norm.setSizePolicy(sizePolicy)
        self.label_only_norm.setMinimumSize(QSize(80, 26))

        self.horizontalLayout_33.addWidget(self.label_only_norm)

        self.OnlyNormcheckBox = QCheckBox(self.QuantizeOptionGroupBox)
        self.OnlyNormcheckBox.setObjectName(u"OnlyNormcheckBox")
        sizePolicy3.setHeightForWidth(self.OnlyNormcheckBox.sizePolicy().hasHeightForWidth())
        self.OnlyNormcheckBox.setSizePolicy(sizePolicy3)

        self.horizontalLayout_33.addWidget(self.OnlyNormcheckBox)

        self.horizontalSpacer_36 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_33.addItem(self.horizontalSpacer_36)


        self.gridLayout_3.addLayout(self.horizontalLayout_33, 1, 0, 1, 1)

        self.horizontalLayout_36 = QHBoxLayout()
        self.horizontalLayout_36.setObjectName(u"horizontalLayout_36")
        self.label_verbose = QLabel(self.QuantizeOptionGroupBox)
        self.label_verbose.setObjectName(u"label_verbose")
        self.label_verbose.setMinimumSize(QSize(80, 26))

        self.horizontalLayout_36.addWidget(self.label_verbose)

        self.VerboseCheckBox = QCheckBox(self.QuantizeOptionGroupBox)
        self.VerboseCheckBox.setObjectName(u"VerboseCheckBox")
        sizePolicy3.setHeightForWidth(self.VerboseCheckBox.sizePolicy().hasHeightForWidth())
        self.VerboseCheckBox.setSizePolicy(sizePolicy3)
        self.VerboseCheckBox.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.VerboseCheckBox.setChecked(False)
        self.VerboseCheckBox.setTristate(False)

        self.horizontalLayout_36.addWidget(self.VerboseCheckBox)

        self.horizontalSpacer_39 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_36.addItem(self.horizontalSpacer_39)


        self.gridLayout_3.addLayout(self.horizontalLayout_36, 1, 3, 1, 1)

        self.horizontalLayout_32 = QHBoxLayout()
        self.horizontalLayout_32.setObjectName(u"horizontalLayout_32")
        self.label_dump_ftmp = QLabel(self.QuantizeOptionGroupBox)
        self.label_dump_ftmp.setObjectName(u"label_dump_ftmp")
        self.label_dump_ftmp.setMinimumSize(QSize(80, 26))

        self.horizontalLayout_32.addWidget(self.label_dump_ftmp)

        self.DumpFtmpCheckBox = QCheckBox(self.QuantizeOptionGroupBox)
        self.DumpFtmpCheckBox.setObjectName(u"DumpFtmpCheckBox")
        sizePolicy3.setHeightForWidth(self.DumpFtmpCheckBox.sizePolicy().hasHeightForWidth())
        self.DumpFtmpCheckBox.setSizePolicy(sizePolicy3)
        self.DumpFtmpCheckBox.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.DumpFtmpCheckBox.setChecked(False)
        self.DumpFtmpCheckBox.setTristate(False)

        self.horizontalLayout_32.addWidget(self.DumpFtmpCheckBox)

        self.horizontalSpacer_40 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_32.addItem(self.horizontalSpacer_40)


        self.gridLayout_3.addLayout(self.horizontalLayout_32, 0, 2, 1, 1)

        self.gridLayout_3.setColumnStretch(0, 1)
        self.gridLayout_3.setColumnStretch(1, 1)
        self.gridLayout_3.setColumnStretch(2, 1)
        self.gridLayout_3.setColumnStretch(3, 1)

        self.verticalLayout_7.addLayout(self.gridLayout_3)

        self.horizontalLayout_37 = QHBoxLayout()
        self.horizontalLayout_37.setObjectName(u"horizontalLayout_37")
        self.label_names_file = QLabel(self.QuantizeOptionGroupBox)
        self.label_names_file.setObjectName(u"label_names_file")
        self.label_names_file.setMinimumSize(QSize(80, 0))

        self.horizontalLayout_37.addWidget(self.label_names_file)

        self.NamesFileLineEdit = QLineEdit(self.QuantizeOptionGroupBox)
        self.NamesFileLineEdit.setObjectName(u"NamesFileLineEdit")

        self.horizontalLayout_37.addWidget(self.NamesFileLineEdit)


        self.verticalLayout_7.addLayout(self.horizontalLayout_37)

        self.horizontalLayout_38 = QHBoxLayout()
        self.horizontalLayout_38.setObjectName(u"horizontalLayout_38")
        self.label_ftmp_csv = QLabel(self.QuantizeOptionGroupBox)
        self.label_ftmp_csv.setObjectName(u"label_ftmp_csv")
        self.label_ftmp_csv.setMinimumSize(QSize(80, 0))

        self.horizontalLayout_38.addWidget(self.label_ftmp_csv)

        self.FtmpCsvLineEdit = QLineEdit(self.QuantizeOptionGroupBox)
        self.FtmpCsvLineEdit.setObjectName(u"FtmpCsvLineEdit")

        self.horizontalLayout_38.addWidget(self.FtmpCsvLineEdit)


        self.verticalLayout_7.addLayout(self.horizontalLayout_38)

        self.horizontalLayout_39 = QHBoxLayout()
        self.horizontalLayout_39.setObjectName(u"horizontalLayout_39")
        self.label_raw_csv = QLabel(self.QuantizeOptionGroupBox)
        self.label_raw_csv.setObjectName(u"label_raw_csv")
        self.label_raw_csv.setMinimumSize(QSize(80, 0))

        self.horizontalLayout_39.addWidget(self.label_raw_csv)

        self.RawCsvLineEdit = QLineEdit(self.QuantizeOptionGroupBox)
        self.RawCsvLineEdit.setObjectName(u"RawCsvLineEdit")

        self.horizontalLayout_39.addWidget(self.RawCsvLineEdit)


        self.verticalLayout_7.addLayout(self.horizontalLayout_39)

        self.horizontalLayout_40 = QHBoxLayout()
        self.horizontalLayout_40.setObjectName(u"horizontalLayout_40")
        self.label_decode_dll = QLabel(self.QuantizeOptionGroupBox)
        self.label_decode_dll.setObjectName(u"label_decode_dll")
        self.label_decode_dll.setMinimumSize(QSize(80, 0))

        self.horizontalLayout_40.addWidget(self.label_decode_dll)

        self.DecodeDllLineEdit = QLineEdit(self.QuantizeOptionGroupBox)
        self.DecodeDllLineEdit.setObjectName(u"DecodeDllLineEdit")

        self.horizontalLayout_40.addWidget(self.DecodeDllLineEdit)


        self.verticalLayout_7.addLayout(self.horizontalLayout_40)


        self.verticalLayout_8.addWidget(self.QuantizeOptionGroupBox)


        self.verticalLayout.addWidget(self.QuantizeGoupBox)

        self.AdaptTargetGroupBox = QGroupBox(self.scrollAreaWidgetContents)
        self.AdaptTargetGroupBox.setObjectName(u"AdaptTargetGroupBox")
        self.AdaptTargetGroupBox.setMinimumSize(QSize(0, 0))
        self.AdaptTargetGroupBox.setTitle(u"[Adapt]")
        self.AdaptTargetGroupBox.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
        self.verticalLayout_10 = QVBoxLayout(self.AdaptTargetGroupBox)
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.horizontalLayout_43 = QHBoxLayout()
        self.horizontalLayout_43.setObjectName(u"horizontalLayout_43")
        self.horizontalLayout_41 = QHBoxLayout()
        self.horizontalLayout_41.setObjectName(u"horizontalLayout_41")
        self.label_adapt_target = QLabel(self.AdaptTargetGroupBox)
        self.label_adapt_target.setObjectName(u"label_adapt_target")

        self.horizontalLayout_41.addWidget(self.label_adapt_target)

        self.AdaptTargetComboBox = QComboBox(self.AdaptTargetGroupBox)
        self.AdaptTargetComboBox.setObjectName(u"AdaptTargetComboBox")

        self.horizontalLayout_41.addWidget(self.AdaptTargetComboBox)

        self.horizontalSpacer_19 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_41.addItem(self.horizontalSpacer_19)


        self.horizontalLayout_43.addLayout(self.horizontalLayout_41)

        self.horizontalLayout_42 = QHBoxLayout()
        self.horizontalLayout_42.setObjectName(u"horizontalLayout_42")
        self.label_adapt_debug = QLabel(self.AdaptTargetGroupBox)
        self.label_adapt_debug.setObjectName(u"label_adapt_debug")

        self.horizontalLayout_42.addWidget(self.label_adapt_debug)

        self.AdaptDebugCheckBox = QCheckBox(self.AdaptTargetGroupBox)
        self.AdaptDebugCheckBox.setObjectName(u"AdaptDebugCheckBox")
        sizePolicy3.setHeightForWidth(self.AdaptDebugCheckBox.sizePolicy().hasHeightForWidth())
        self.AdaptDebugCheckBox.setSizePolicy(sizePolicy3)

        self.horizontalLayout_42.addWidget(self.AdaptDebugCheckBox)


        self.horizontalLayout_43.addLayout(self.horizontalLayout_42)

        self.horizontalSpacer_12 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_43.addItem(self.horizontalSpacer_12)

        self.horizontalLayout_43.setStretch(0, 1)
        self.horizontalLayout_43.setStretch(1, 1)
        self.horizontalLayout_43.setStretch(2, 2)

        self.verticalLayout_10.addLayout(self.horizontalLayout_43)

        self.AdaptOptionToolButton = QToolButton(self.AdaptTargetGroupBox)
        self.AdaptOptionToolButton.setObjectName(u"AdaptOptionToolButton")
        sizePolicy2.setHeightForWidth(self.AdaptOptionToolButton.sizePolicy().hasHeightForWidth())
        self.AdaptOptionToolButton.setSizePolicy(sizePolicy2)
        self.AdaptOptionToolButton.setCursor(QCursor(Qt.ArrowCursor))
        self.AdaptOptionToolButton.setMouseTracking(False)
        self.AdaptOptionToolButton.setTabletTracking(False)
        self.AdaptOptionToolButton.setStyleSheet(u" QToolButton {\n"
"                text-align: center;  /* \u4f7f\u6587\u672c\u5c45\u4e2d */\n"
"                padding-right: 16px;  /* \u786e\u4fdd\u7bad\u5934\u548c\u6587\u672c\u6709\u9002\u5f53\u7684\u95f4\u8ddd */\n"
"            }")
        self.AdaptOptionToolButton.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.AdaptOptionToolButton.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.AdaptOptionToolButton.setAutoRaise(False)
        self.AdaptOptionToolButton.setArrowType(Qt.ArrowType.RightArrow)

        self.verticalLayout_10.addWidget(self.AdaptOptionToolButton)

        self.AdaptOptionGroupBox = QGroupBox(self.AdaptTargetGroupBox)
        self.AdaptOptionGroupBox.setObjectName(u"AdaptOptionGroupBox")
        sizePolicy2.setHeightForWidth(self.AdaptOptionGroupBox.sizePolicy().hasHeightForWidth())
        self.AdaptOptionGroupBox.setSizePolicy(sizePolicy2)
        self.AdaptOptionGroupBox.setMinimumSize(QSize(0, 0))
        self.AdaptOptionGroupBox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.AdaptOptionGroupBox.setFlat(False)
        self.AdaptOptionGroupBox.setCheckable(False)
        self.verticalLayout_9 = QVBoxLayout(self.AdaptOptionGroupBox)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.verticalLayout_9.setContentsMargins(-1, 9, -1, -1)
        self.horizontalLayout_44 = QHBoxLayout()
        self.horizontalLayout_44.setObjectName(u"horizontalLayout_44")
        self.label_adapt_customop = QLabel(self.AdaptOptionGroupBox)
        self.label_adapt_customop.setObjectName(u"label_adapt_customop")
        sizePolicy2.setHeightForWidth(self.label_adapt_customop.sizePolicy().hasHeightForWidth())
        self.label_adapt_customop.setSizePolicy(sizePolicy2)
        self.label_adapt_customop.setMinimumSize(QSize(117, 26))

        self.horizontalLayout_44.addWidget(self.label_adapt_customop)

        self.AdaptCustomOpInput = QLineEdit(self.AdaptOptionGroupBox)
        self.AdaptCustomOpInput.setObjectName(u"AdaptCustomOpInput")
        self.AdaptCustomOpInput.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_44.addWidget(self.AdaptCustomOpInput)


        self.verticalLayout_9.addLayout(self.horizontalLayout_44)

        self.horizontalLayout_53 = QHBoxLayout()
        self.horizontalLayout_53.setObjectName(u"horizontalLayout_53")
        self.label_adapt_config = QLabel(self.AdaptOptionGroupBox)
        self.label_adapt_config.setObjectName(u"label_adapt_config")
        self.label_adapt_config.setMinimumSize(QSize(117, 26))

        self.horizontalLayout_53.addWidget(self.label_adapt_config)

        self.AdaptConfigLineEdit = QLineEdit(self.AdaptOptionGroupBox)
        self.AdaptConfigLineEdit.setObjectName(u"AdaptConfigLineEdit")
        self.AdaptConfigLineEdit.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_53.addWidget(self.AdaptConfigLineEdit)


        self.verticalLayout_9.addLayout(self.horizontalLayout_53)

        self.horizontalLayout_45 = QHBoxLayout()
        self.horizontalLayout_45.setObjectName(u"horizontalLayout_45")
        self.horizontalLayout_46 = QHBoxLayout()
        self.horizontalLayout_46.setObjectName(u"horizontalLayout_46")
        self.label_adapt_customop_on = QLabel(self.AdaptOptionGroupBox)
        self.label_adapt_customop_on.setObjectName(u"label_adapt_customop_on")
        sizePolicy.setHeightForWidth(self.label_adapt_customop_on.sizePolicy().hasHeightForWidth())
        self.label_adapt_customop_on.setSizePolicy(sizePolicy)
        self.label_adapt_customop_on.setMinimumSize(QSize(117, 26))

        self.horizontalLayout_46.addWidget(self.label_adapt_customop_on)

        self.AdaptCustomOpOnInput = QLineEdit(self.AdaptOptionGroupBox)
        self.AdaptCustomOpOnInput.setObjectName(u"AdaptCustomOpOnInput")
        self.AdaptCustomOpOnInput.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_46.addWidget(self.AdaptCustomOpOnInput)


        self.horizontalLayout_45.addLayout(self.horizontalLayout_46)

        self.horizontalLayout_47 = QHBoxLayout()
        self.horizontalLayout_47.setObjectName(u"horizontalLayout_47")
        self.label_adapt_pass_on = QLabel(self.AdaptOptionGroupBox)
        self.label_adapt_pass_on.setObjectName(u"label_adapt_pass_on")
        sizePolicy.setHeightForWidth(self.label_adapt_pass_on.sizePolicy().hasHeightForWidth())
        self.label_adapt_pass_on.setSizePolicy(sizePolicy)
        self.label_adapt_pass_on.setMinimumSize(QSize(75, 26))

        self.horizontalLayout_47.addWidget(self.label_adapt_pass_on)

        self.AdaptPassOnInput = QLineEdit(self.AdaptOptionGroupBox)
        self.AdaptPassOnInput.setObjectName(u"AdaptPassOnInput")
        self.AdaptPassOnInput.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_47.addWidget(self.AdaptPassOnInput)


        self.horizontalLayout_45.addLayout(self.horizontalLayout_47)

        self.horizontalLayout_48 = QHBoxLayout()
        self.horizontalLayout_48.setObjectName(u"horizontalLayout_48")
        self.label_adapt_pass_off = QLabel(self.AdaptOptionGroupBox)
        self.label_adapt_pass_off.setObjectName(u"label_adapt_pass_off")
        sizePolicy.setHeightForWidth(self.label_adapt_pass_off.sizePolicy().hasHeightForWidth())
        self.label_adapt_pass_off.setSizePolicy(sizePolicy)
        self.label_adapt_pass_off.setMinimumSize(QSize(75, 26))

        self.horizontalLayout_48.addWidget(self.label_adapt_pass_off)

        self.AdaptPassOffInput = QLineEdit(self.AdaptOptionGroupBox)
        self.AdaptPassOffInput.setObjectName(u"AdaptPassOffInput")
        self.AdaptPassOffInput.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_48.addWidget(self.AdaptPassOffInput)


        self.horizontalLayout_45.addLayout(self.horizontalLayout_48)


        self.verticalLayout_9.addLayout(self.horizontalLayout_45)


        self.verticalLayout_10.addWidget(self.AdaptOptionGroupBox)


        self.verticalLayout.addWidget(self.AdaptTargetGroupBox)

        self.GenerateGroupBox = QGroupBox(self.scrollAreaWidgetContents)
        self.GenerateGroupBox.setObjectName(u"GenerateGroupBox")
        self.GenerateGroupBox.setMinimumSize(QSize(0, 0))
        self.GenerateGroupBox.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
        self.verticalLayout_12 = QVBoxLayout(self.GenerateGroupBox)
        self.verticalLayout_12.setObjectName(u"verticalLayout_12")
        self.GenerateOptionToolButton = QToolButton(self.GenerateGroupBox)
        self.GenerateOptionToolButton.setObjectName(u"GenerateOptionToolButton")
        sizePolicy2.setHeightForWidth(self.GenerateOptionToolButton.sizePolicy().hasHeightForWidth())
        self.GenerateOptionToolButton.setSizePolicy(sizePolicy2)
        self.GenerateOptionToolButton.setCursor(QCursor(Qt.ArrowCursor))
        self.GenerateOptionToolButton.setMouseTracking(False)
        self.GenerateOptionToolButton.setTabletTracking(False)
        self.GenerateOptionToolButton.setStyleSheet(u" QToolButton {\n"
"                text-align: center;  /* \u4f7f\u6587\u672c\u5c45\u4e2d */\n"
"                padding-right: 16px;  /* \u786e\u4fdd\u7bad\u5934\u548c\u6587\u672c\u6709\u9002\u5f53\u7684\u95f4\u8ddd */\n"
"            }")
        self.GenerateOptionToolButton.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.GenerateOptionToolButton.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.GenerateOptionToolButton.setAutoRaise(False)
        self.GenerateOptionToolButton.setArrowType(Qt.ArrowType.RightArrow)

        self.verticalLayout_12.addWidget(self.GenerateOptionToolButton)

        self.GenerateOptionGroupBox = QGroupBox(self.GenerateGroupBox)
        self.GenerateOptionGroupBox.setObjectName(u"GenerateOptionGroupBox")
        sizePolicy2.setHeightForWidth(self.GenerateOptionGroupBox.sizePolicy().hasHeightForWidth())
        self.GenerateOptionGroupBox.setSizePolicy(sizePolicy2)
        self.GenerateOptionGroupBox.setMinimumSize(QSize(0, 0))
        self.GenerateOptionGroupBox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.GenerateOptionGroupBox.setFlat(False)
        self.GenerateOptionGroupBox.setCheckable(False)
        self.verticalLayout_11 = QVBoxLayout(self.GenerateOptionGroupBox)
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.horizontalLayout_49 = QHBoxLayout()
        self.horizontalLayout_49.setObjectName(u"horizontalLayout_49")
        self.label_generate_ddr_base = QLabel(self.GenerateOptionGroupBox)
        self.label_generate_ddr_base.setObjectName(u"label_generate_ddr_base")
        sizePolicy2.setHeightForWidth(self.label_generate_ddr_base.sizePolicy().hasHeightForWidth())
        self.label_generate_ddr_base.setSizePolicy(sizePolicy2)
        self.label_generate_ddr_base.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_49.addWidget(self.label_generate_ddr_base)

        self.GenerateDDRBaseInput = QLineEdit(self.GenerateOptionGroupBox)
        self.GenerateDDRBaseInput.setObjectName(u"GenerateDDRBaseInput")
        self.GenerateDDRBaseInput.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_49.addWidget(self.GenerateDDRBaseInput)


        self.verticalLayout_11.addLayout(self.horizontalLayout_49)

        self.horizontalLayout_58 = QHBoxLayout()
        self.horizontalLayout_58.setObjectName(u"horizontalLayout_58")
        self.horizontalLayout_50 = QHBoxLayout()
        self.horizontalLayout_50.setObjectName(u"horizontalLayout_50")
        self.label_generate_rows = QLabel(self.GenerateOptionGroupBox)
        self.label_generate_rows.setObjectName(u"label_generate_rows")
        self.label_generate_rows.setMinimumSize(QSize(62, 26))

        self.horizontalLayout_50.addWidget(self.label_generate_rows)

        self.GenerateRowsSpinBox = QSpinBox(self.GenerateOptionGroupBox)
        self.GenerateRowsSpinBox.setObjectName(u"GenerateRowsSpinBox")
        self.GenerateRowsSpinBox.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_50.addWidget(self.GenerateRowsSpinBox)

        self.horizontalSpacer_15 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_50.addItem(self.horizontalSpacer_15)


        self.horizontalLayout_58.addLayout(self.horizontalLayout_50)

        self.horizontalLayout_51 = QHBoxLayout()
        self.horizontalLayout_51.setObjectName(u"horizontalLayout_51")
        self.label_generate_cols = QLabel(self.GenerateOptionGroupBox)
        self.label_generate_cols.setObjectName(u"label_generate_cols")
        self.label_generate_cols.setMinimumSize(QSize(62, 26))

        self.horizontalLayout_51.addWidget(self.label_generate_cols)

        self.GenerateColsSpinBox = QSpinBox(self.GenerateOptionGroupBox)
        self.GenerateColsSpinBox.setObjectName(u"GenerateColsSpinBox")
        self.GenerateColsSpinBox.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_51.addWidget(self.GenerateColsSpinBox)

        self.horizontalSpacer_16 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_51.addItem(self.horizontalSpacer_16)


        self.horizontalLayout_58.addLayout(self.horizontalLayout_51)

        self.horizontalLayout_52 = QHBoxLayout()
        self.horizontalLayout_52.setObjectName(u"horizontalLayout_52")
        self.label_generate_xlmopt = QLabel(self.GenerateOptionGroupBox)
        self.label_generate_xlmopt.setObjectName(u"label_generate_xlmopt")
        self.label_generate_xlmopt.setMinimumSize(QSize(62, 26))

        self.horizontalLayout_52.addWidget(self.label_generate_xlmopt)

        self.GenerateXlmoptspinBox = QSpinBox(self.GenerateOptionGroupBox)
        self.GenerateXlmoptspinBox.setObjectName(u"GenerateXlmoptspinBox")
        self.GenerateXlmoptspinBox.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_52.addWidget(self.GenerateXlmoptspinBox)

        self.horizontalSpacer_17 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_52.addItem(self.horizontalSpacer_17)


        self.horizontalLayout_58.addLayout(self.horizontalLayout_52)

        self.horizontalLayout_54 = QHBoxLayout()
        self.horizontalLayout_54.setObjectName(u"horizontalLayout_54")
        self.label_generate_klmopt = QLabel(self.GenerateOptionGroupBox)
        self.label_generate_klmopt.setObjectName(u"label_generate_klmopt")
        self.label_generate_klmopt.setMinimumSize(QSize(62, 26))

        self.horizontalLayout_54.addWidget(self.label_generate_klmopt)

        self.GenerateKlmoptSpinBox = QSpinBox(self.GenerateOptionGroupBox)
        self.GenerateKlmoptSpinBox.setObjectName(u"GenerateKlmoptSpinBox")
        self.GenerateKlmoptSpinBox.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_54.addWidget(self.GenerateKlmoptSpinBox)

        self.horizontalSpacer_18 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_54.addItem(self.horizontalSpacer_18)


        self.horizontalLayout_58.addLayout(self.horizontalLayout_54)


        self.verticalLayout_11.addLayout(self.horizontalLayout_58)

        self.horizontalLayout_59 = QHBoxLayout()
        self.horizontalLayout_59.setObjectName(u"horizontalLayout_59")
        self.horizontalLayout_55 = QHBoxLayout()
        self.horizontalLayout_55.setObjectName(u"horizontalLayout_55")
        self.label_generate_icropt = QLabel(self.GenerateOptionGroupBox)
        self.label_generate_icropt.setObjectName(u"label_generate_icropt")
        self.label_generate_icropt.setMinimumSize(QSize(62, 26))

        self.horizontalLayout_55.addWidget(self.label_generate_icropt)

        self.GenerateIcroptComboBox = QComboBox(self.GenerateOptionGroupBox)
        self.GenerateIcroptComboBox.setObjectName(u"GenerateIcroptComboBox")
        self.GenerateIcroptComboBox.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_55.addWidget(self.GenerateIcroptComboBox)

        self.horizontalSpacer_13 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_55.addItem(self.horizontalSpacer_13)


        self.horizontalLayout_59.addLayout(self.horizontalLayout_55)

        self.horizontalLayout_56 = QHBoxLayout()
        self.horizontalLayout_56.setObjectName(u"horizontalLayout_56")
        self.label_generate_ocmopt = QLabel(self.GenerateOptionGroupBox)
        self.label_generate_ocmopt.setObjectName(u"label_generate_ocmopt")
        self.label_generate_ocmopt.setMinimumSize(QSize(62, 26))

        self.horizontalLayout_56.addWidget(self.label_generate_ocmopt)

        self.GenerateOcmoptComboBox = QComboBox(self.GenerateOptionGroupBox)
        self.GenerateOcmoptComboBox.setObjectName(u"GenerateOcmoptComboBox")
        self.GenerateOcmoptComboBox.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_56.addWidget(self.GenerateOcmoptComboBox)

        self.horizontalSpacer_14 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_56.addItem(self.horizontalSpacer_14)


        self.horizontalLayout_59.addLayout(self.horizontalLayout_56)

        self.horizontalLayout_57 = QHBoxLayout()
        self.horizontalLayout_57.setObjectName(u"horizontalLayout_57")
        self.label_generate_version = QLabel(self.GenerateOptionGroupBox)
        self.label_generate_version.setObjectName(u"label_generate_version")
        self.label_generate_version.setMinimumSize(QSize(62, 26))

        self.horizontalLayout_57.addWidget(self.label_generate_version)

        self.GenerateVersionCheckBox = QCheckBox(self.GenerateOptionGroupBox)
        self.GenerateVersionCheckBox.setObjectName(u"GenerateVersionCheckBox")
        sizePolicy3.setHeightForWidth(self.GenerateVersionCheckBox.sizePolicy().hasHeightForWidth())
        self.GenerateVersionCheckBox.setSizePolicy(sizePolicy3)
        self.GenerateVersionCheckBox.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_57.addWidget(self.GenerateVersionCheckBox)

        self.horizontalSpacer_42 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_57.addItem(self.horizontalSpacer_42)


        self.horizontalLayout_59.addLayout(self.horizontalLayout_57)

        self.horizontalLayout_59.setStretch(0, 1)
        self.horizontalLayout_59.setStretch(1, 1)
        self.horizontalLayout_59.setStretch(2, 1)

        self.verticalLayout_11.addLayout(self.horizontalLayout_59)


        self.verticalLayout_12.addWidget(self.GenerateOptionGroupBox)


        self.verticalLayout.addWidget(self.GenerateGroupBox)

        self.SimulateGroupBox = QGroupBox(self.scrollAreaWidgetContents)
        self.SimulateGroupBox.setObjectName(u"SimulateGroupBox")
        self.SimulateGroupBox.setMinimumSize(QSize(0, 0))
        self.SimulateGroupBox.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
        self.verticalLayout_14 = QVBoxLayout(self.SimulateGroupBox)
        self.verticalLayout_14.setObjectName(u"verticalLayout_14")
        self.horizontalLayout_72 = QHBoxLayout()
        self.horizontalLayout_72.setObjectName(u"horizontalLayout_72")
        self.horizontalLayout_66 = QHBoxLayout()
        self.horizontalLayout_66.setObjectName(u"horizontalLayout_66")
        self.label_simulate_target = QLabel(self.SimulateGroupBox)
        self.label_simulate_target.setObjectName(u"label_simulate_target")
        self.label_simulate_target.setMinimumSize(QSize(68, 26))

        self.horizontalLayout_66.addWidget(self.label_simulate_target)

        self.SimulateTargetComboBox = QComboBox(self.SimulateGroupBox)
        self.SimulateTargetComboBox.setObjectName(u"SimulateTargetComboBox")
        self.SimulateTargetComboBox.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_66.addWidget(self.SimulateTargetComboBox)

        self.horizontalSpacer_20 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_66.addItem(self.horizontalSpacer_20)


        self.horizontalLayout_72.addLayout(self.horizontalLayout_66)

        self.horizontalLayout_67 = QHBoxLayout()
        self.horizontalLayout_67.setObjectName(u"horizontalLayout_67")
        self.label_simulate_stage = QLabel(self.SimulateGroupBox)
        self.label_simulate_stage.setObjectName(u"label_simulate_stage")
        self.label_simulate_stage.setMinimumSize(QSize(68, 26))

        self.horizontalLayout_67.addWidget(self.label_simulate_stage)

        self.SimulateStageComboBox = QComboBox(self.SimulateGroupBox)
        self.SimulateStageComboBox.setObjectName(u"SimulateStageComboBox")
        self.SimulateStageComboBox.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_67.addWidget(self.SimulateStageComboBox)

        self.horizontalSpacer_21 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_67.addItem(self.horizontalSpacer_21)


        self.horizontalLayout_72.addLayout(self.horizontalLayout_67)

        self.horizontalLayout_68 = QHBoxLayout()
        self.horizontalLayout_68.setObjectName(u"horizontalLayout_68")
        self.label_simulate_fake_qf = QLabel(self.SimulateGroupBox)
        self.label_simulate_fake_qf.setObjectName(u"label_simulate_fake_qf")
        self.label_simulate_fake_qf.setMinimumSize(QSize(68, 26))

        self.horizontalLayout_68.addWidget(self.label_simulate_fake_qf)

        self.FakeQfCheckBox = QCheckBox(self.SimulateGroupBox)
        self.FakeQfCheckBox.setObjectName(u"FakeQfCheckBox")
        sizePolicy3.setHeightForWidth(self.FakeQfCheckBox.sizePolicy().hasHeightForWidth())
        self.FakeQfCheckBox.setSizePolicy(sizePolicy3)
        self.FakeQfCheckBox.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_68.addWidget(self.FakeQfCheckBox)

        self.horizontalSpacer_41 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_68.addItem(self.horizontalSpacer_41)


        self.horizontalLayout_72.addLayout(self.horizontalLayout_68)

        self.horizontalLayout_69 = QHBoxLayout()
        self.horizontalLayout_69.setObjectName(u"horizontalLayout_69")
        self.label_simulate_dump_ftmp = QLabel(self.SimulateGroupBox)
        self.label_simulate_dump_ftmp.setObjectName(u"label_simulate_dump_ftmp")
        self.label_simulate_dump_ftmp.setMinimumSize(QSize(68, 26))

        self.horizontalLayout_69.addWidget(self.label_simulate_dump_ftmp)

        self.SimulateDumpFtmpComboBox = QComboBox(self.SimulateGroupBox)
        self.SimulateDumpFtmpComboBox.setObjectName(u"SimulateDumpFtmpComboBox")
        self.SimulateDumpFtmpComboBox.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_69.addWidget(self.SimulateDumpFtmpComboBox)

        self.horizontalSpacer_22 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_69.addItem(self.horizontalSpacer_22)


        self.horizontalLayout_72.addLayout(self.horizontalLayout_69)

        self.horizontalLayout_72.setStretch(0, 1)
        self.horizontalLayout_72.setStretch(1, 1)
        self.horizontalLayout_72.setStretch(2, 1)
        self.horizontalLayout_72.setStretch(3, 1)

        self.verticalLayout_14.addLayout(self.horizontalLayout_72)

        self.horizontalLayout_70 = QHBoxLayout()
        self.horizontalLayout_70.setObjectName(u"horizontalLayout_70")
        self.label_simulate_names = QLabel(self.SimulateGroupBox)
        self.label_simulate_names.setObjectName(u"label_simulate_names")
        self.label_simulate_names.setMinimumSize(QSize(62, 26))

        self.horizontalLayout_70.addWidget(self.label_simulate_names)

        self.SimulateNamesInput = QLineEdit(self.SimulateGroupBox)
        self.SimulateNamesInput.setObjectName(u"SimulateNamesInput")
        self.SimulateNamesInput.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_70.addWidget(self.SimulateNamesInput)


        self.verticalLayout_14.addLayout(self.horizontalLayout_70)

        self.horizontalLayout_71 = QHBoxLayout()
        self.horizontalLayout_71.setObjectName(u"horizontalLayout_71")
        self.label_simulate_image = QLabel(self.SimulateGroupBox)
        self.label_simulate_image.setObjectName(u"label_simulate_image")
        self.label_simulate_image.setMinimumSize(QSize(62, 26))

        self.horizontalLayout_71.addWidget(self.label_simulate_image)

        self.SimulateImageInput = QLineEdit(self.SimulateGroupBox)
        self.SimulateImageInput.setObjectName(u"SimulateImageInput")
        self.SimulateImageInput.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_71.addWidget(self.SimulateImageInput)


        self.verticalLayout_14.addLayout(self.horizontalLayout_71)

        self.SimulateOptionToolButton = QToolButton(self.SimulateGroupBox)
        self.SimulateOptionToolButton.setObjectName(u"SimulateOptionToolButton")
        sizePolicy2.setHeightForWidth(self.SimulateOptionToolButton.sizePolicy().hasHeightForWidth())
        self.SimulateOptionToolButton.setSizePolicy(sizePolicy2)
        self.SimulateOptionToolButton.setCursor(QCursor(Qt.ArrowCursor))
        self.SimulateOptionToolButton.setMouseTracking(False)
        self.SimulateOptionToolButton.setTabletTracking(False)
        self.SimulateOptionToolButton.setStyleSheet(u" QToolButton {\n"
"                text-align: center;  /* \u4f7f\u6587\u672c\u5c45\u4e2d */\n"
"                padding-right: 16px;  /* \u786e\u4fdd\u7bad\u5934\u548c\u6587\u672c\u6709\u9002\u5f53\u7684\u95f4\u8ddd */\n"
"            }")
        self.SimulateOptionToolButton.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.SimulateOptionToolButton.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.SimulateOptionToolButton.setAutoRaise(False)
        self.SimulateOptionToolButton.setArrowType(Qt.ArrowType.RightArrow)

        self.verticalLayout_14.addWidget(self.SimulateOptionToolButton)

        self.SimulateOptionGroupBox = QGroupBox(self.SimulateGroupBox)
        self.SimulateOptionGroupBox.setObjectName(u"SimulateOptionGroupBox")
        sizePolicy2.setHeightForWidth(self.SimulateOptionGroupBox.sizePolicy().hasHeightForWidth())
        self.SimulateOptionGroupBox.setSizePolicy(sizePolicy2)
        self.SimulateOptionGroupBox.setMinimumSize(QSize(0, 0))
        self.SimulateOptionGroupBox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.SimulateOptionGroupBox.setFlat(False)
        self.SimulateOptionGroupBox.setCheckable(False)
        self.verticalLayout_13 = QVBoxLayout(self.SimulateOptionGroupBox)
        self.verticalLayout_13.setObjectName(u"verticalLayout_13")
        self.horizontalLayout_93 = QHBoxLayout()
        self.horizontalLayout_93.setObjectName(u"horizontalLayout_93")
        self.horizontalLayout_75 = QHBoxLayout()
        self.horizontalLayout_75.setObjectName(u"horizontalLayout_75")
        self.label_simulate_log_time = QLabel(self.SimulateOptionGroupBox)
        self.label_simulate_log_time.setObjectName(u"label_simulate_log_time")
        sizePolicy2.setHeightForWidth(self.label_simulate_log_time.sizePolicy().hasHeightForWidth())
        self.label_simulate_log_time.setSizePolicy(sizePolicy2)
        self.label_simulate_log_time.setMinimumSize(QSize(91, 26))

        self.horizontalLayout_75.addWidget(self.label_simulate_log_time)

        self.SimulateLogTimeCheckBox = QCheckBox(self.SimulateOptionGroupBox)
        self.SimulateLogTimeCheckBox.setObjectName(u"SimulateLogTimeCheckBox")
        sizePolicy3.setHeightForWidth(self.SimulateLogTimeCheckBox.sizePolicy().hasHeightForWidth())
        self.SimulateLogTimeCheckBox.setSizePolicy(sizePolicy3)
        self.SimulateLogTimeCheckBox.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_75.addWidget(self.SimulateLogTimeCheckBox)

        self.horizontalSpacer_24 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_75.addItem(self.horizontalSpacer_24)


        self.horizontalLayout_93.addLayout(self.horizontalLayout_75)

        self.horizontalLayout_76 = QHBoxLayout()
        self.horizontalLayout_76.setObjectName(u"horizontalLayout_76")
        self.label_simulate_log_io = QLabel(self.SimulateOptionGroupBox)
        self.label_simulate_log_io.setObjectName(u"label_simulate_log_io")
        sizePolicy2.setHeightForWidth(self.label_simulate_log_io.sizePolicy().hasHeightForWidth())
        self.label_simulate_log_io.setSizePolicy(sizePolicy2)
        self.label_simulate_log_io.setMinimumSize(QSize(91, 26))

        self.horizontalLayout_76.addWidget(self.label_simulate_log_io)

        self.SimulateLogIOCheckBox = QCheckBox(self.SimulateOptionGroupBox)
        self.SimulateLogIOCheckBox.setObjectName(u"SimulateLogIOCheckBox")
        sizePolicy3.setHeightForWidth(self.SimulateLogIOCheckBox.sizePolicy().hasHeightForWidth())
        self.SimulateLogIOCheckBox.setSizePolicy(sizePolicy3)
        self.SimulateLogIOCheckBox.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_76.addWidget(self.SimulateLogIOCheckBox)

        self.horizontalSpacer_25 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_76.addItem(self.horizontalSpacer_25)


        self.horizontalLayout_93.addLayout(self.horizontalLayout_76)

        self.horizontalLayout_77 = QHBoxLayout()
        self.horizontalLayout_77.setObjectName(u"horizontalLayout_77")
        self.label_simulate_show = QLabel(self.SimulateOptionGroupBox)
        self.label_simulate_show.setObjectName(u"label_simulate_show")
        sizePolicy2.setHeightForWidth(self.label_simulate_show.sizePolicy().hasHeightForWidth())
        self.label_simulate_show.setSizePolicy(sizePolicy2)
        self.label_simulate_show.setMinimumSize(QSize(91, 26))

        self.horizontalLayout_77.addWidget(self.label_simulate_show)

        self.SimulateShowCheckBox = QCheckBox(self.SimulateOptionGroupBox)
        self.SimulateShowCheckBox.setObjectName(u"SimulateShowCheckBox")
        sizePolicy3.setHeightForWidth(self.SimulateShowCheckBox.sizePolicy().hasHeightForWidth())
        self.SimulateShowCheckBox.setSizePolicy(sizePolicy3)
        self.SimulateShowCheckBox.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_77.addWidget(self.SimulateShowCheckBox)

        self.horizontalSpacer_26 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_77.addItem(self.horizontalSpacer_26)


        self.horizontalLayout_93.addLayout(self.horizontalLayout_77)

        self.horizontalLayout_78 = QHBoxLayout()
        self.horizontalLayout_78.setObjectName(u"horizontalLayout_78")
        self.label_simulate_save = QLabel(self.SimulateOptionGroupBox)
        self.label_simulate_save.setObjectName(u"label_simulate_save")
        sizePolicy2.setHeightForWidth(self.label_simulate_save.sizePolicy().hasHeightForWidth())
        self.label_simulate_save.setSizePolicy(sizePolicy2)
        self.label_simulate_save.setMinimumSize(QSize(91, 26))

        self.horizontalLayout_78.addWidget(self.label_simulate_save)

        self.SimulateSaveCheckBox = QCheckBox(self.SimulateOptionGroupBox)
        self.SimulateSaveCheckBox.setObjectName(u"SimulateSaveCheckBox")
        sizePolicy3.setHeightForWidth(self.SimulateSaveCheckBox.sizePolicy().hasHeightForWidth())
        self.SimulateSaveCheckBox.setSizePolicy(sizePolicy3)
        self.SimulateSaveCheckBox.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_78.addWidget(self.SimulateSaveCheckBox)

        self.horizontalSpacer_27 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_78.addItem(self.horizontalSpacer_27)


        self.horizontalLayout_93.addLayout(self.horizontalLayout_78)


        self.verticalLayout_13.addLayout(self.horizontalLayout_93)

        self.horizontalLayout_85 = QHBoxLayout()
        self.horizontalLayout_85.setObjectName(u"horizontalLayout_85")
        self.horizontalLayout_80 = QHBoxLayout()
        self.horizontalLayout_80.setObjectName(u"horizontalLayout_80")
        self.label_simulate_logCalcTime = QLabel(self.SimulateOptionGroupBox)
        self.label_simulate_logCalcTime.setObjectName(u"label_simulate_logCalcTime")
        sizePolicy2.setHeightForWidth(self.label_simulate_logCalcTime.sizePolicy().hasHeightForWidth())
        self.label_simulate_logCalcTime.setSizePolicy(sizePolicy2)
        self.label_simulate_logCalcTime.setMinimumSize(QSize(91, 26))

        self.horizontalLayout_80.addWidget(self.label_simulate_logCalcTime)

        self.SimulateLogCalcTimeCheckBox = QCheckBox(self.SimulateOptionGroupBox)
        self.SimulateLogCalcTimeCheckBox.setObjectName(u"SimulateLogCalcTimeCheckBox")
        sizePolicy3.setHeightForWidth(self.SimulateLogCalcTimeCheckBox.sizePolicy().hasHeightForWidth())
        self.SimulateLogCalcTimeCheckBox.setSizePolicy(sizePolicy3)
        self.SimulateLogCalcTimeCheckBox.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_80.addWidget(self.SimulateLogCalcTimeCheckBox)

        self.horizontalSpacer_28 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_80.addItem(self.horizontalSpacer_28)


        self.horizontalLayout_85.addLayout(self.horizontalLayout_80)

        self.horizontalLayout_81 = QHBoxLayout()
        self.horizontalLayout_81.setObjectName(u"horizontalLayout_81")
        self.label_simulate_log_io_3 = QLabel(self.SimulateOptionGroupBox)
        self.label_simulate_log_io_3.setObjectName(u"label_simulate_log_io_3")
        sizePolicy2.setHeightForWidth(self.label_simulate_log_io_3.sizePolicy().hasHeightForWidth())
        self.label_simulate_log_io_3.setSizePolicy(sizePolicy2)
        self.label_simulate_log_io_3.setMinimumSize(QSize(91, 26))

        self.horizontalLayout_81.addWidget(self.label_simulate_log_io_3)

        self.SimulateLogIOInfoCheckBox = QCheckBox(self.SimulateOptionGroupBox)
        self.SimulateLogIOInfoCheckBox.setObjectName(u"SimulateLogIOInfoCheckBox")
        sizePolicy3.setHeightForWidth(self.SimulateLogIOInfoCheckBox.sizePolicy().hasHeightForWidth())
        self.SimulateLogIOInfoCheckBox.setSizePolicy(sizePolicy3)
        self.SimulateLogIOInfoCheckBox.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_81.addWidget(self.SimulateLogIOInfoCheckBox)

        self.horizontalSpacer_29 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_81.addItem(self.horizontalSpacer_29)


        self.horizontalLayout_85.addLayout(self.horizontalLayout_81)

        self.horizontalLayout_82 = QHBoxLayout()
        self.horizontalLayout_82.setObjectName(u"horizontalLayout_82")
        self.label_simulate_dump_weights = QLabel(self.SimulateOptionGroupBox)
        self.label_simulate_dump_weights.setObjectName(u"label_simulate_dump_weights")
        sizePolicy2.setHeightForWidth(self.label_simulate_dump_weights.sizePolicy().hasHeightForWidth())
        self.label_simulate_dump_weights.setSizePolicy(sizePolicy2)
        self.label_simulate_dump_weights.setMinimumSize(QSize(91, 26))

        self.horizontalLayout_82.addWidget(self.label_simulate_dump_weights)

        self.SimulateDumpWeightsCheckBox = QCheckBox(self.SimulateOptionGroupBox)
        self.SimulateDumpWeightsCheckBox.setObjectName(u"SimulateDumpWeightsCheckBox")
        sizePolicy3.setHeightForWidth(self.SimulateDumpWeightsCheckBox.sizePolicy().hasHeightForWidth())
        self.SimulateDumpWeightsCheckBox.setSizePolicy(sizePolicy3)
        self.SimulateDumpWeightsCheckBox.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_82.addWidget(self.SimulateDumpWeightsCheckBox)

        self.horizontalSpacer_30 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_82.addItem(self.horizontalSpacer_30)


        self.horizontalLayout_85.addLayout(self.horizontalLayout_82)

        self.horizontalLayout_83 = QHBoxLayout()
        self.horizontalLayout_83.setObjectName(u"horizontalLayout_83")
        self.label_simulate_dump_bias = QLabel(self.SimulateOptionGroupBox)
        self.label_simulate_dump_bias.setObjectName(u"label_simulate_dump_bias")
        sizePolicy2.setHeightForWidth(self.label_simulate_dump_bias.sizePolicy().hasHeightForWidth())
        self.label_simulate_dump_bias.setSizePolicy(sizePolicy2)
        self.label_simulate_dump_bias.setMinimumSize(QSize(91, 26))

        self.horizontalLayout_83.addWidget(self.label_simulate_dump_bias)

        self.SimulateDumpBiasCheckBox = QCheckBox(self.SimulateOptionGroupBox)
        self.SimulateDumpBiasCheckBox.setObjectName(u"SimulateDumpBiasCheckBox")
        sizePolicy3.setHeightForWidth(self.SimulateDumpBiasCheckBox.sizePolicy().hasHeightForWidth())
        self.SimulateDumpBiasCheckBox.setSizePolicy(sizePolicy3)
        self.SimulateDumpBiasCheckBox.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_83.addWidget(self.SimulateDumpBiasCheckBox)

        self.horizontalSpacer_31 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_83.addItem(self.horizontalSpacer_31)


        self.horizontalLayout_85.addLayout(self.horizontalLayout_83)


        self.verticalLayout_13.addLayout(self.horizontalLayout_85)

        self.horizontalLayout_86 = QHBoxLayout()
        self.horizontalLayout_86.setObjectName(u"horizontalLayout_86")
        self.horizontalLayout_84 = QHBoxLayout()
        self.horizontalLayout_84.setObjectName(u"horizontalLayout_84")
        self.label_simulate_cudamode = QLabel(self.SimulateOptionGroupBox)
        self.label_simulate_cudamode.setObjectName(u"label_simulate_cudamode")
        sizePolicy2.setHeightForWidth(self.label_simulate_cudamode.sizePolicy().hasHeightForWidth())
        self.label_simulate_cudamode.setSizePolicy(sizePolicy2)
        self.label_simulate_cudamode.setMinimumSize(QSize(91, 26))

        self.horizontalLayout_84.addWidget(self.label_simulate_cudamode)

        self.SimulateCudaModeCheckBox = QCheckBox(self.SimulateOptionGroupBox)
        self.SimulateCudaModeCheckBox.setObjectName(u"SimulateCudaModeCheckBox")
        sizePolicy3.setHeightForWidth(self.SimulateCudaModeCheckBox.sizePolicy().hasHeightForWidth())
        self.SimulateCudaModeCheckBox.setSizePolicy(sizePolicy3)
        self.SimulateCudaModeCheckBox.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_84.addWidget(self.SimulateCudaModeCheckBox)

        self.horizontalSpacer_32 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_84.addItem(self.horizontalSpacer_32)


        self.horizontalLayout_86.addLayout(self.horizontalLayout_84)

        self.horizontalLayout_79 = QHBoxLayout()
        self.horizontalLayout_79.setObjectName(u"horizontalLayout_79")
        self.label_simulate_ebytes = QLabel(self.SimulateOptionGroupBox)
        self.label_simulate_ebytes.setObjectName(u"label_simulate_ebytes")
        sizePolicy2.setHeightForWidth(self.label_simulate_ebytes.sizePolicy().hasHeightForWidth())
        self.label_simulate_ebytes.setSizePolicy(sizePolicy2)
        self.label_simulate_ebytes.setMinimumSize(QSize(91, 26))

        self.horizontalLayout_79.addWidget(self.label_simulate_ebytes)

        self.SimulateEbytesComboBox = QComboBox(self.SimulateOptionGroupBox)
        self.SimulateEbytesComboBox.setObjectName(u"SimulateEbytesComboBox")
        self.SimulateEbytesComboBox.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_79.addWidget(self.SimulateEbytesComboBox)

        self.horizontalSpacer_33 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_79.addItem(self.horizontalSpacer_33)


        self.horizontalLayout_86.addLayout(self.horizontalLayout_79)

        self.horizontalSpacer_23 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_86.addItem(self.horizontalSpacer_23)

        self.horizontalLayout_86.setStretch(0, 1)
        self.horizontalLayout_86.setStretch(1, 1)
        self.horizontalLayout_86.setStretch(2, 2)

        self.verticalLayout_13.addLayout(self.horizontalLayout_86)

        self.horizontalLayout_94 = QHBoxLayout()
        self.horizontalLayout_94.setObjectName(u"horizontalLayout_94")
        self.horizontalLayout_89 = QHBoxLayout()
        self.horizontalLayout_89.setObjectName(u"horizontalLayout_89")
        self.label_simulate_dump_ftmp_start = QLabel(self.SimulateOptionGroupBox)
        self.label_simulate_dump_ftmp_start.setObjectName(u"label_simulate_dump_ftmp_start")
        sizePolicy2.setHeightForWidth(self.label_simulate_dump_ftmp_start.sizePolicy().hasHeightForWidth())
        self.label_simulate_dump_ftmp_start.setSizePolicy(sizePolicy2)
        self.label_simulate_dump_ftmp_start.setMinimumSize(QSize(116, 26))

        self.horizontalLayout_89.addWidget(self.label_simulate_dump_ftmp_start)

        self.SimulateDumpFtmpStartInput = QLineEdit(self.SimulateOptionGroupBox)
        self.SimulateDumpFtmpStartInput.setObjectName(u"SimulateDumpFtmpStartInput")
        self.SimulateDumpFtmpStartInput.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_89.addWidget(self.SimulateDumpFtmpStartInput)


        self.horizontalLayout_94.addLayout(self.horizontalLayout_89)

        self.horizontalLayout_91 = QHBoxLayout()
        self.horizontalLayout_91.setObjectName(u"horizontalLayout_91")
        self.label_simulate_dump_ftmp_end = QLabel(self.SimulateOptionGroupBox)
        self.label_simulate_dump_ftmp_end.setObjectName(u"label_simulate_dump_ftmp_end")
        sizePolicy2.setHeightForWidth(self.label_simulate_dump_ftmp_end.sizePolicy().hasHeightForWidth())
        self.label_simulate_dump_ftmp_end.setSizePolicy(sizePolicy2)
        self.label_simulate_dump_ftmp_end.setMinimumSize(QSize(116, 26))

        self.horizontalLayout_91.addWidget(self.label_simulate_dump_ftmp_end)

        self.SimulateDumpFtmpEndInput = QLineEdit(self.SimulateOptionGroupBox)
        self.SimulateDumpFtmpEndInput.setObjectName(u"SimulateDumpFtmpEndInput")
        self.SimulateDumpFtmpEndInput.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_91.addWidget(self.SimulateDumpFtmpEndInput)


        self.horizontalLayout_94.addLayout(self.horizontalLayout_91)


        self.verticalLayout_13.addLayout(self.horizontalLayout_94)

        self.horizontalLayout_95 = QHBoxLayout()
        self.horizontalLayout_95.setObjectName(u"horizontalLayout_95")
        self.horizontalLayout_87 = QHBoxLayout()
        self.horizontalLayout_87.setObjectName(u"horizontalLayout_87")
        self.label_simulate_dump_bias_start = QLabel(self.SimulateOptionGroupBox)
        self.label_simulate_dump_bias_start.setObjectName(u"label_simulate_dump_bias_start")
        sizePolicy2.setHeightForWidth(self.label_simulate_dump_bias_start.sizePolicy().hasHeightForWidth())
        self.label_simulate_dump_bias_start.setSizePolicy(sizePolicy2)
        self.label_simulate_dump_bias_start.setMinimumSize(QSize(116, 26))

        self.horizontalLayout_87.addWidget(self.label_simulate_dump_bias_start)

        self.SimulateDumpBiasStartInput = QLineEdit(self.SimulateOptionGroupBox)
        self.SimulateDumpBiasStartInput.setObjectName(u"SimulateDumpBiasStartInput")
        self.SimulateDumpBiasStartInput.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_87.addWidget(self.SimulateDumpBiasStartInput)


        self.horizontalLayout_95.addLayout(self.horizontalLayout_87)

        self.horizontalLayout_88 = QHBoxLayout()
        self.horizontalLayout_88.setObjectName(u"horizontalLayout_88")
        self.label_simulate_dump_bias_end = QLabel(self.SimulateOptionGroupBox)
        self.label_simulate_dump_bias_end.setObjectName(u"label_simulate_dump_bias_end")
        sizePolicy2.setHeightForWidth(self.label_simulate_dump_bias_end.sizePolicy().hasHeightForWidth())
        self.label_simulate_dump_bias_end.setSizePolicy(sizePolicy2)
        self.label_simulate_dump_bias_end.setMinimumSize(QSize(116, 26))

        self.horizontalLayout_88.addWidget(self.label_simulate_dump_bias_end)

        self.SimulateDumpBiasEndInput = QLineEdit(self.SimulateOptionGroupBox)
        self.SimulateDumpBiasEndInput.setObjectName(u"SimulateDumpBiasEndInput")
        self.SimulateDumpBiasEndInput.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_88.addWidget(self.SimulateDumpBiasEndInput)


        self.horizontalLayout_95.addLayout(self.horizontalLayout_88)


        self.verticalLayout_13.addLayout(self.horizontalLayout_95)

        self.horizontalLayout_96 = QHBoxLayout()
        self.horizontalLayout_96.setObjectName(u"horizontalLayout_96")
        self.horizontalLayout_90 = QHBoxLayout()
        self.horizontalLayout_90.setObjectName(u"horizontalLayout_90")
        self.label_simulate_dump_weights_start = QLabel(self.SimulateOptionGroupBox)
        self.label_simulate_dump_weights_start.setObjectName(u"label_simulate_dump_weights_start")
        sizePolicy2.setHeightForWidth(self.label_simulate_dump_weights_start.sizePolicy().hasHeightForWidth())
        self.label_simulate_dump_weights_start.setSizePolicy(sizePolicy2)
        self.label_simulate_dump_weights_start.setMinimumSize(QSize(116, 26))

        self.horizontalLayout_90.addWidget(self.label_simulate_dump_weights_start)

        self.SimulateDumpWeightsStartInput = QLineEdit(self.SimulateOptionGroupBox)
        self.SimulateDumpWeightsStartInput.setObjectName(u"SimulateDumpWeightsStartInput")
        self.SimulateDumpWeightsStartInput.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_90.addWidget(self.SimulateDumpWeightsStartInput)


        self.horizontalLayout_96.addLayout(self.horizontalLayout_90)

        self.horizontalLayout_92 = QHBoxLayout()
        self.horizontalLayout_92.setObjectName(u"horizontalLayout_92")
        self.label_simulate_dump_weights_end = QLabel(self.SimulateOptionGroupBox)
        self.label_simulate_dump_weights_end.setObjectName(u"label_simulate_dump_weights_end")
        sizePolicy2.setHeightForWidth(self.label_simulate_dump_weights_end.sizePolicy().hasHeightForWidth())
        self.label_simulate_dump_weights_end.setSizePolicy(sizePolicy2)
        self.label_simulate_dump_weights_end.setMinimumSize(QSize(116, 26))

        self.horizontalLayout_92.addWidget(self.label_simulate_dump_weights_end)

        self.SimulateDumpWeightsEndInput = QLineEdit(self.SimulateOptionGroupBox)
        self.SimulateDumpWeightsEndInput.setObjectName(u"SimulateDumpWeightsEndInput")
        self.SimulateDumpWeightsEndInput.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_92.addWidget(self.SimulateDumpWeightsEndInput)


        self.horizontalLayout_96.addLayout(self.horizontalLayout_92)


        self.verticalLayout_13.addLayout(self.horizontalLayout_96)

        self.horizontalLayout_64 = QHBoxLayout()
        self.horizontalLayout_64.setObjectName(u"horizontalLayout_64")
        self.horizontalLayout_60 = QHBoxLayout()
        self.horizontalLayout_60.setObjectName(u"horizontalLayout_60")
        self.label_simulate_names_path = QLabel(self.SimulateOptionGroupBox)
        self.label_simulate_names_path.setObjectName(u"label_simulate_names_path")
        sizePolicy2.setHeightForWidth(self.label_simulate_names_path.sizePolicy().hasHeightForWidth())
        self.label_simulate_names_path.setSizePolicy(sizePolicy2)
        self.label_simulate_names_path.setMinimumSize(QSize(116, 26))

        self.horizontalLayout_60.addWidget(self.label_simulate_names_path)

        self.SimulateNamesPathInput = QLineEdit(self.SimulateOptionGroupBox)
        self.SimulateNamesPathInput.setObjectName(u"SimulateNamesPathInput")
        self.SimulateNamesPathInput.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_60.addWidget(self.SimulateNamesPathInput)


        self.horizontalLayout_64.addLayout(self.horizontalLayout_60)

        self.horizontalLayout_61 = QHBoxLayout()
        self.horizontalLayout_61.setObjectName(u"horizontalLayout_61")
        self.label_simulate_decode_dll = QLabel(self.SimulateOptionGroupBox)
        self.label_simulate_decode_dll.setObjectName(u"label_simulate_decode_dll")
        sizePolicy2.setHeightForWidth(self.label_simulate_decode_dll.sizePolicy().hasHeightForWidth())
        self.label_simulate_decode_dll.setSizePolicy(sizePolicy2)
        self.label_simulate_decode_dll.setMinimumSize(QSize(116, 26))

        self.horizontalLayout_61.addWidget(self.label_simulate_decode_dll)

        self.SimulateDecodeDllInput = QLineEdit(self.SimulateOptionGroupBox)
        self.SimulateDecodeDllInput.setObjectName(u"SimulateDecodeDllInput")
        self.SimulateDecodeDllInput.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_61.addWidget(self.SimulateDecodeDllInput)


        self.horizontalLayout_64.addLayout(self.horizontalLayout_61)


        self.verticalLayout_13.addLayout(self.horizontalLayout_64)

        self.horizontalLayout_65 = QHBoxLayout()
        self.horizontalLayout_65.setObjectName(u"horizontalLayout_65")
        self.horizontalLayout_62 = QHBoxLayout()
        self.horizontalLayout_62.setObjectName(u"horizontalLayout_62")
        self.label_simulate_post_method = QLabel(self.SimulateOptionGroupBox)
        self.label_simulate_post_method.setObjectName(u"label_simulate_post_method")
        sizePolicy2.setHeightForWidth(self.label_simulate_post_method.sizePolicy().hasHeightForWidth())
        self.label_simulate_post_method.setSizePolicy(sizePolicy2)
        self.label_simulate_post_method.setMinimumSize(QSize(116, 26))

        self.horizontalLayout_62.addWidget(self.label_simulate_post_method)

        self.SimulatePostMethodInput = QLineEdit(self.SimulateOptionGroupBox)
        self.SimulatePostMethodInput.setObjectName(u"SimulatePostMethodInput")
        self.SimulatePostMethodInput.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_62.addWidget(self.SimulatePostMethodInput)


        self.horizontalLayout_65.addLayout(self.horizontalLayout_62)

        self.horizontalLayout_63 = QHBoxLayout()
        self.horizontalLayout_63.setObjectName(u"horizontalLayout_63")
        self.label_simulate_ff_backend = QLabel(self.SimulateOptionGroupBox)
        self.label_simulate_ff_backend.setObjectName(u"label_simulate_ff_backend")
        sizePolicy2.setHeightForWidth(self.label_simulate_ff_backend.sizePolicy().hasHeightForWidth())
        self.label_simulate_ff_backend.setSizePolicy(sizePolicy2)
        self.label_simulate_ff_backend.setMinimumSize(QSize(116, 26))

        self.horizontalLayout_63.addWidget(self.label_simulate_ff_backend)

        self.SimulateFFBackendInput = QLineEdit(self.SimulateOptionGroupBox)
        self.SimulateFFBackendInput.setObjectName(u"SimulateFFBackendInput")
        self.SimulateFFBackendInput.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_63.addWidget(self.SimulateFFBackendInput)


        self.horizontalLayout_65.addLayout(self.horizontalLayout_63)


        self.verticalLayout_13.addLayout(self.horizontalLayout_65)


        self.verticalLayout_14.addWidget(self.SimulateOptionGroupBox)


        self.verticalLayout.addWidget(self.SimulateGroupBox)

        self.ConfigGroupBox = QGroupBox(self.scrollAreaWidgetContents)
        self.ConfigGroupBox.setObjectName(u"ConfigGroupBox")
        self.ConfigGroupBox.setMinimumSize(QSize(0, 0))
        self.ConfigGroupBox.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
        self.verticalLayout_15 = QVBoxLayout(self.ConfigGroupBox)
        self.verticalLayout_15.setObjectName(u"verticalLayout_15")
        self.horizontalLayout_73 = QHBoxLayout()
        self.horizontalLayout_73.setObjectName(u"horizontalLayout_73")
        self.label_config_customop_on = QLabel(self.ConfigGroupBox)
        self.label_config_customop_on.setObjectName(u"label_config_customop_on")
        sizePolicy.setHeightForWidth(self.label_config_customop_on.sizePolicy().hasHeightForWidth())
        self.label_config_customop_on.setSizePolicy(sizePolicy)
        self.label_config_customop_on.setMinimumSize(QSize(117, 26))

        self.horizontalLayout_73.addWidget(self.label_config_customop_on)

        self.ConfigCustomOpOnInput = QLineEdit(self.ConfigGroupBox)
        self.ConfigCustomOpOnInput.setObjectName(u"ConfigCustomOpOnInput")
        self.ConfigCustomOpOnInput.setMinimumSize(QSize(0, 26))

        self.horizontalLayout_73.addWidget(self.ConfigCustomOpOnInput)


        self.verticalLayout_15.addLayout(self.horizontalLayout_73)

        self.horizontalLayout_74 = QHBoxLayout()
        self.horizontalLayout_74.setObjectName(u"horizontalLayout_74")
        self.label_config_customop_config = QLabel(self.ConfigGroupBox)
        self.label_config_customop_config.setObjectName(u"label_config_customop_config")
        sizePolicy2.setHeightForWidth(self.label_config_customop_config.sizePolicy().hasHeightForWidth())
        self.label_config_customop_config.setSizePolicy(sizePolicy2)
        self.label_config_customop_config.setMinimumSize(QSize(117, 0))

        self.horizontalLayout_74.addWidget(self.label_config_customop_config)

        self.ConfigCustomopConfigInput = QLineEdit(self.ConfigGroupBox)
        self.ConfigCustomopConfigInput.setObjectName(u"ConfigCustomopConfigInput")

        self.horizontalLayout_74.addWidget(self.ConfigCustomopConfigInput)


        self.verticalLayout_15.addLayout(self.horizontalLayout_74)


        self.verticalLayout.addWidget(self.ConfigGroupBox)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.verticalLayout_3.addWidget(self.scrollArea)

        self.horizontalLayout_23 = QHBoxLayout()
        self.horizontalLayout_23.setObjectName(u"horizontalLayout_23")
        self.horizontalSpacer_7 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_23.addItem(self.horizontalSpacer_7)

        self.ConfirmButton = QPushButton(ConfigDialog)
        self.ConfirmButton.setObjectName(u"ConfirmButton")
        sizePolicy2.setHeightForWidth(self.ConfirmButton.sizePolicy().hasHeightForWidth())
        self.ConfirmButton.setSizePolicy(sizePolicy2)
        self.ConfirmButton.setMinimumSize(QSize(60, 30))
        self.ConfirmButton.setMaximumSize(QSize(60, 30))

        self.horizontalLayout_23.addWidget(self.ConfirmButton)


        self.verticalLayout_3.addLayout(self.horizontalLayout_23)

        QWidget.setTabOrder(self.scrollArea, self.NetworkInput)
        QWidget.setTabOrder(self.NetworkInput, self.ChooseNetworkButton)
        QWidget.setTabOrder(self.ChooseNetworkButton, self.NameInput)
        QWidget.setTabOrder(self.NameInput, self.FrameWorkComboBox)
        QWidget.setTabOrder(self.FrameWorkComboBox, self.FrameVersionComboBox)
        QWidget.setTabOrder(self.FrameVersionComboBox, self.PostMethodComboBox)
        QWidget.setTabOrder(self.PostMethodComboBox, self.PreMethodComboBox)
        QWidget.setTabOrder(self.PreMethodComboBox, self.ShapeInput)
        QWidget.setTabOrder(self.ShapeInput, self.InputFormatComboBox)
        QWidget.setTabOrder(self.InputFormatComboBox, self.ChannelSwapInput)
        QWidget.setTabOrder(self.ChannelSwapInput, self.PreMeanInput)
        QWidget.setTabOrder(self.PreMeanInput, self.PreScaleInput)
        QWidget.setTabOrder(self.PreScaleInput, self.WeightInput)
        QWidget.setTabOrder(self.WeightInput, self.CustomOpInput)
        QWidget.setTabOrder(self.CustomOpInput, self.DebugCheckBox)
        QWidget.setTabOrder(self.DebugCheckBox, self.TargetComboBox)
        QWidget.setTabOrder(self.TargetComboBox, self.CustomopConfigOpInput)
        QWidget.setTabOrder(self.CustomopConfigOpInput, self.ConfigInput)
        QWidget.setTabOrder(self.ConfigInput, self.CustomOpOnInput)
        QWidget.setTabOrder(self.CustomOpOnInput, self.PassOnInput)
        QWidget.setTabOrder(self.PassOnInput, self.PassOffInput)

        self.retranslateUi(ConfigDialog)

        QMetaObject.connectSlotsByName(ConfigDialog)
    # setupUi

    def retranslateUi(self, ConfigDialog):
        ConfigDialog.setWindowTitle(QCoreApplication.translate("ConfigDialog", u"\u53c2\u6570\u914d\u7f6e", None))
        self.ParseGroupBox.setTitle(QCoreApplication.translate("ConfigDialog", u"[Parse]", None))
        self.label_input.setText(QCoreApplication.translate("ConfigDialog", u"input", None))
        self.label_weights.setText(QCoreApplication.translate("ConfigDialog", u"weights", None))
        self.label_framework.setText(QCoreApplication.translate("ConfigDialog", u"framework", None))
        self.label_custom_op.setText(QCoreApplication.translate("ConfigDialog", u"custom_op", None))
        self.label_pre_method.setText(QCoreApplication.translate("ConfigDialog", u"pre_method", None))
        self.label_name.setText(QCoreApplication.translate("ConfigDialog", u"name", None))
        self.label_network.setText(QCoreApplication.translate("ConfigDialog", u"network", None))
        self.ChooseNetworkButton.setText(QCoreApplication.translate("ConfigDialog", u"\u9009\u62e9", None))
        self.label_pre_scale.setText(QCoreApplication.translate("ConfigDialog", u"pre_scale", None))
        self.label_pre_mean.setText(QCoreApplication.translate("ConfigDialog", u"pre_mean", None))
        self.label_chann_swap.setText(QCoreApplication.translate("ConfigDialog", u"chann_swap", None))
        self.label_input_format.setText(QCoreApplication.translate("ConfigDialog", u"input_format", None))
        self.label_frame_version.setText(QCoreApplication.translate("ConfigDialog", u"frame_version", None))
        self.label_post_method.setText(QCoreApplication.translate("ConfigDialog", u"post_method", None))
        self.OptimizeGroupBox.setTitle(QCoreApplication.translate("ConfigDialog", u"[Optimize]", None))
        self.label_debug.setText(QCoreApplication.translate("ConfigDialog", u"debug", None))
        self.DebugCheckBox.setText("")
        self.label_target.setText(QCoreApplication.translate("ConfigDialog", u"target", None))
        self.OptimizeOptionToolButton.setText(QCoreApplication.translate("ConfigDialog", u"\u53ef\u9009", None))
        self.OptimizeOptionGroupBox.setTitle(QCoreApplication.translate("ConfigDialog", u"\u53ef\u9009", None))
        self.label_customop_config.setText(QCoreApplication.translate("ConfigDialog", u"customop_config", None))
        self.label_config.setText(QCoreApplication.translate("ConfigDialog", u"config", None))
        self.label_customop_on.setText(QCoreApplication.translate("ConfigDialog", u"customop_on", None))
        self.label_pass_on.setText(QCoreApplication.translate("ConfigDialog", u"pass_on", None))
        self.label_pass_off.setText(QCoreApplication.translate("ConfigDialog", u"pass_off", None))
        self.QuantizeGoupBox.setTitle(QCoreApplication.translate("ConfigDialog", u"[Quantize]", None))
        self.label_forward_list.setText(QCoreApplication.translate("ConfigDialog", u"forward_list", None))
        self.label_bits.setText(QCoreApplication.translate("ConfigDialog", u"bits", None))
        self.label_per.setText(QCoreApplication.translate("ConfigDialog", u"per", None))
        self.label_forward_dir.setText(QCoreApplication.translate("ConfigDialog", u"forward_dir", None))
        self.label_saturation.setText(QCoreApplication.translate("ConfigDialog", u"saturation", None))
        self.label_forward_mode.setText(QCoreApplication.translate("ConfigDialog", u"forward\n"
"mode", None))
        self.QuantizeOptionToolButton.setText(QCoreApplication.translate("ConfigDialog", u"\u53ef\u9009", None))
        self.QuantizeOptionGroupBox.setTitle(QCoreApplication.translate("ConfigDialog", u"\u53ef\u9009", None))
        self.label_batch.setText(QCoreApplication.translate("ConfigDialog", u"batch", None))
        self.label_no_transinput.setText(QCoreApplication.translate("ConfigDialog", u"no_transinput", None))
        self.NoTransinputcheckBox.setText("")
        self.label_bin_num.setText(QCoreApplication.translate("ConfigDialog", u"bin_num", None))
        self.label_before_relu.setText(QCoreApplication.translate("ConfigDialog", u"before_relu", None))
        self.BeforeReluCheckBox.setText("")
        self.label_only_norm.setText(QCoreApplication.translate("ConfigDialog", u"only_norm", None))
        self.OnlyNormcheckBox.setText("")
        self.label_verbose.setText(QCoreApplication.translate("ConfigDialog", u"verbose", None))
        self.VerboseCheckBox.setText("")
        self.label_dump_ftmp.setText(QCoreApplication.translate("ConfigDialog", u"dump_ftmp", None))
        self.DumpFtmpCheckBox.setText("")
        self.label_names_file.setText(QCoreApplication.translate("ConfigDialog", u"names_file", None))
        self.label_ftmp_csv.setText(QCoreApplication.translate("ConfigDialog", u"ftmp_csv", None))
        self.label_raw_csv.setText(QCoreApplication.translate("ConfigDialog", u"raw_csv", None))
        self.label_decode_dll.setText(QCoreApplication.translate("ConfigDialog", u"decode_dll", None))
        self.label_adapt_target.setText(QCoreApplication.translate("ConfigDialog", u"target", None))
        self.label_adapt_debug.setText(QCoreApplication.translate("ConfigDialog", u"debug", None))
        self.AdaptDebugCheckBox.setText("")
        self.AdaptOptionToolButton.setText(QCoreApplication.translate("ConfigDialog", u"\u53ef\u9009", None))
        self.AdaptOptionGroupBox.setTitle(QCoreApplication.translate("ConfigDialog", u"\u53ef\u9009", None))
        self.label_adapt_customop.setText(QCoreApplication.translate("ConfigDialog", u"customop_config", None))
        self.label_adapt_config.setText(QCoreApplication.translate("ConfigDialog", u"config", None))
        self.label_adapt_customop_on.setText(QCoreApplication.translate("ConfigDialog", u"customop_on", None))
        self.label_adapt_pass_on.setText(QCoreApplication.translate("ConfigDialog", u"pass_on", None))
        self.label_adapt_pass_off.setText(QCoreApplication.translate("ConfigDialog", u"pass_off", None))
        self.GenerateGroupBox.setTitle(QCoreApplication.translate("ConfigDialog", u"[Generate]", None))
        self.GenerateOptionToolButton.setText(QCoreApplication.translate("ConfigDialog", u"\u53ef\u9009", None))
        self.GenerateOptionGroupBox.setTitle(QCoreApplication.translate("ConfigDialog", u"\u53ef\u9009", None))
        self.label_generate_ddr_base.setText(QCoreApplication.translate("ConfigDialog", u"ddr_base", None))
        self.label_generate_rows.setText(QCoreApplication.translate("ConfigDialog", u"rows", None))
        self.label_generate_cols.setText(QCoreApplication.translate("ConfigDialog", u"cols", None))
        self.label_generate_xlmopt.setText(QCoreApplication.translate("ConfigDialog", u"xlmopt", None))
        self.label_generate_klmopt.setText(QCoreApplication.translate("ConfigDialog", u"klmopt", None))
        self.label_generate_icropt.setText(QCoreApplication.translate("ConfigDialog", u"icropt", None))
        self.label_generate_ocmopt.setText(QCoreApplication.translate("ConfigDialog", u"ocmopt", None))
        self.label_generate_version.setText(QCoreApplication.translate("ConfigDialog", u"version", None))
        self.GenerateVersionCheckBox.setText("")
        self.SimulateGroupBox.setTitle(QCoreApplication.translate("ConfigDialog", u"[Simulate]", None))
        self.label_simulate_target.setText(QCoreApplication.translate("ConfigDialog", u"target", None))
        self.label_simulate_stage.setText(QCoreApplication.translate("ConfigDialog", u"stage", None))
        self.label_simulate_fake_qf.setText(QCoreApplication.translate("ConfigDialog", u"fake_qf", None))
        self.FakeQfCheckBox.setText("")
        self.label_simulate_dump_ftmp.setText(QCoreApplication.translate("ConfigDialog", u"dump_ftmp", None))
        self.label_simulate_names.setText(QCoreApplication.translate("ConfigDialog", u"names", None))
        self.label_simulate_image.setText(QCoreApplication.translate("ConfigDialog", u"image", None))
        self.SimulateOptionToolButton.setText(QCoreApplication.translate("ConfigDialog", u"\u53ef\u9009", None))
        self.SimulateOptionGroupBox.setTitle(QCoreApplication.translate("ConfigDialog", u"\u53ef\u9009", None))
        self.label_simulate_log_time.setText(QCoreApplication.translate("ConfigDialog", u"log_time", None))
        self.SimulateLogTimeCheckBox.setText("")
        self.label_simulate_log_io.setText(QCoreApplication.translate("ConfigDialog", u"log_io", None))
        self.SimulateLogIOCheckBox.setText("")
        self.label_simulate_show.setText(QCoreApplication.translate("ConfigDialog", u"show", None))
        self.SimulateShowCheckBox.setText("")
        self.label_simulate_save.setText(QCoreApplication.translate("ConfigDialog", u"save", None))
        self.SimulateSaveCheckBox.setText("")
        self.label_simulate_logCalcTime.setText(QCoreApplication.translate("ConfigDialog", u"logCalcTime", None))
        self.SimulateLogCalcTimeCheckBox.setText("")
        self.label_simulate_log_io_3.setText(QCoreApplication.translate("ConfigDialog", u"logIOInfo", None))
        self.SimulateLogIOInfoCheckBox.setText("")
        self.label_simulate_dump_weights.setText(QCoreApplication.translate("ConfigDialog", u"dump_weights", None))
        self.SimulateDumpWeightsCheckBox.setText("")
        self.label_simulate_dump_bias.setText(QCoreApplication.translate("ConfigDialog", u"dump_bias", None))
        self.SimulateDumpBiasCheckBox.setText("")
        self.label_simulate_cudamode.setText(QCoreApplication.translate("ConfigDialog", u"cudamode", None))
        self.SimulateCudaModeCheckBox.setText("")
        self.label_simulate_ebytes.setText(QCoreApplication.translate("ConfigDialog", u"ebytes", None))
        self.label_simulate_dump_ftmp_start.setText(QCoreApplication.translate("ConfigDialog", u"dump_ftmp_start", None))
        self.label_simulate_dump_ftmp_end.setText(QCoreApplication.translate("ConfigDialog", u"dump_ftmp_end", None))
        self.label_simulate_dump_bias_start.setText(QCoreApplication.translate("ConfigDialog", u"dump_bias_start", None))
        self.label_simulate_dump_bias_end.setText(QCoreApplication.translate("ConfigDialog", u"dump_bias_end", None))
        self.label_simulate_dump_weights_start.setText(QCoreApplication.translate("ConfigDialog", u"dump_weights_start", None))
        self.label_simulate_dump_weights_end.setText(QCoreApplication.translate("ConfigDialog", u"dump_weights_end", None))
        self.label_simulate_names_path.setText(QCoreApplication.translate("ConfigDialog", u"names_path", None))
        self.label_simulate_decode_dll.setText(QCoreApplication.translate("ConfigDialog", u"decode_dll", None))
        self.label_simulate_post_method.setText(QCoreApplication.translate("ConfigDialog", u"post_method", None))
        self.label_simulate_ff_backend.setText(QCoreApplication.translate("ConfigDialog", u"ff_backend", None))
        self.ConfigGroupBox.setTitle(QCoreApplication.translate("ConfigDialog", u"[Config]", None))
        self.label_config_customop_on.setText(QCoreApplication.translate("ConfigDialog", u"customop_on", None))
        self.label_config_customop_config.setText(QCoreApplication.translate("ConfigDialog", u"customop_config", None))
        self.ConfirmButton.setText(QCoreApplication.translate("ConfigDialog", u"\u786e\u8ba4", None))
    # retranslateUi

