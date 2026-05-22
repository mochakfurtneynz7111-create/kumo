# qt_compat.py
import sys

try:
    from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton, 
                                   QLineEdit, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, 
                                   QScrollArea, QTextEdit, QCheckBox, QGroupBox, QSpinBox, 
                                   QStatusBar, QToolBar, QMenuBar, QMenu, QMessageBox,
                                   QToolButton, QFileDialog, QFileSystemModel, QHeaderView, QRadioButton,
                                   QComboBox, QDialog, QSpacerItem, QSizePolicy, QGridLayout,
                                   QDial, QDialog, QDoubleSpinBox, QPlainTextEdit
    )
    from PySide6.QtCore import (
        Qt, QObject, QSize, QPoint, QRect, 
        QUrl, QThread, Signal, Slot, QProcess,
        QProcessEnvironment, QEvent, QMutex, QTimer
    )
    from PySide6.QtGui import (QIcon, QStandardItemModel, QStandardItem,
    )
    from PySide6.QtWebEngineWidgets import (
        QWebEngineView
    )
    PYSIDE_VERSION = 6
except ImportError:
    from PySide2.QtWidgets import *
    from PySide2.QtCore import *
    from PySide2.QtGui import *
    from PySide2.QtWebEngineWidgets import *
    PYSIDE_VERSION = 2

# 处理枚举值差异（Qt6使用Flag后缀）
if PYSIDE_VERSION == 6:
    Alignment = Qt.AlignmentFlag
else:
    Alignment = Qt

__all__ = ['PYSIDE_VERSION', 'Alignment'] + [n for n in globals() if n.startswith('Q')]

