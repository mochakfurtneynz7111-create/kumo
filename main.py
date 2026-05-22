import sys
from src.logic.GetSrcPath import resource_path
from src.gui.main_window import MainWindow

from src.QTCompat import QApplication
# from qt_material import apply_stylesheet
# from PySide6.QtCore import Qt
# from netron import server
import os

if __name__ == "__main__":
    # QCoreApplication.setAttribute
    # QCoreApplication.setAttribute(Qt::AA_UseSoftwareOpenGL)
    app = QApplication(sys.argv)
    window = MainWindow()
    # window.showMaximized()
    # 从文件加载样式表
    with open(resource_path("style/style.qss"), "r", encoding="utf-8") as file:
        app.setStyleSheet(file.read())  # 读取文件内容并应用样式表
    # apply_stylesheet(app, theme='light_blue.xml')
    # 显示窗口
    window.show()
    if hasattr(app, 'exec'):
        sys.exit(app.exec())  # PySide6/PyQt6
    else:
        sys.exit(app.exec_())  # PySide2/PyQt5


  
