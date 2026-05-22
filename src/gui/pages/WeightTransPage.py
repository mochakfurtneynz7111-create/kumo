from src.QTCompat import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
                              QPushButton, QFileDialog, QScrollArea, QMessageBox)
from src.QTCompat import Qt, Slot
import os
import logging
from src.QTCompat import Signal, QSize, Qt, QProcess, QProcessEnvironment,QMutex

logger = logging.getLogger(__name__)

class WeightTransPage(QWidget):
    def __init__(self):
        super().__init__()
        self.process = None  # 进程对象
        self.initUI()
        
    def initUI(self):
        mainLayout = QVBoxLayout(self)
        mainLayout.setContentsMargins(9, 6, 9, 6)
        mainLayout.setSpacing(12)
        mainLayout.setAlignment(Qt.AlignTop)
        
        # Python解释器路径
        pythonPathLayout = QHBoxLayout()
        pythonPathLabel = QLabel("Python解释器:")
        pythonPathLabel.setMinimumWidth(120)
        pythonPathLayout.addWidget(pythonPathLabel)
        self.pythonPathEdit = QLineEdit()
        pythonPathBrowseBtn = QPushButton("浏览...")
        pythonPathBrowseBtn.clicked.connect(lambda: self.browseOpenFile(self.pythonPathEdit))
        pythonPathLayout.addWidget(self.pythonPathEdit)
        pythonPathLayout.addWidget(pythonPathBrowseBtn)
        mainLayout.addLayout(pythonPathLayout)

        # 待转换的模型路径
        modelPathLayout = QHBoxLayout()
        modelPathLabel = QLabel("待转换模型路径:")
        modelPathLabel.setMinimumWidth(120)
        modelPathLayout.addWidget(modelPathLabel)
        self.modelPathEdit = QLineEdit()
        modelPathBrowseBtn = QPushButton("浏览...")
        modelPathBrowseBtn.clicked.connect(lambda: self.browseOpenFile(self.modelPathEdit))
        modelPathLayout.addWidget(self.modelPathEdit)
        modelPathLayout.addWidget(modelPathBrowseBtn)
        mainLayout.addLayout(modelPathLayout)
        
        # 模型转换脚本路径
        scriptPathLayout = QHBoxLayout()
        scriptPathLabel = QLabel("转换脚本路径:")
        scriptPathLabel.setMinimumWidth(120)
        scriptPathLayout.addWidget(scriptPathLabel)
        self.scriptPathEdit = QLineEdit()
        scriptPathBrowseBtn = QPushButton("浏览...")
        scriptPathBrowseBtn.clicked.connect(lambda: self.browseOpenFile(self.scriptPathEdit))
        scriptPathLayout.addWidget(self.scriptPathEdit)
        scriptPathLayout.addWidget(scriptPathBrowseBtn)
        mainLayout.addLayout(scriptPathLayout)
        
        # 脚本工作目录
        workDirLayout = QHBoxLayout()
        workDirLabel = QLabel("脚本工作目录:")
        workDirLabel.setMinimumWidth(120)
        workDirLayout.addWidget(workDirLabel)
        self.workDirEdit = QLineEdit()
        workDirBrowseBtn = QPushButton("浏览...")
        workDirBrowseBtn.clicked.connect(lambda: self.browseOpenPath(self.workDirEdit))
        workDirLayout.addWidget(self.workDirEdit)
        workDirLayout.addWidget(workDirBrowseBtn)
        mainLayout.addLayout(workDirLayout)

        # 转换后模型保存目录
        saveDirLayout = QHBoxLayout()
        saveDirLabel = QLabel("保存目录:")
        saveDirLabel.setMinimumWidth(120)
        saveDirLayout.addWidget(saveDirLabel)
        self.saveDirEdit = QLineEdit()
        saveDirBrowseBtn = QPushButton("浏览...")
        saveDirBrowseBtn.clicked.connect(lambda: self.browseOpenPath(self.saveDirEdit))
        saveDirLayout.addWidget(self.saveDirEdit)
        saveDirLayout.addWidget(saveDirBrowseBtn)
        mainLayout.addLayout(saveDirLayout)
        
        # 开始转换按钮
        btnLayout = QHBoxLayout()
        btnLayout.setAlignment(Qt.AlignRight)
        self.convertBtn = QPushButton("开始转换")
        self.convertBtn.setMinimumWidth(120)
        self.convertBtn.clicked.connect(self.startConversion)
        btnLayout.addWidget(self.convertBtn)
        mainLayout.addLayout(btnLayout)
        
        self.setLayout(mainLayout)
    
    def browseOpenFile(self, editUI):
        """浏览打开文件"""
        oldFile = editUI.text()
        if os.path.exists(oldFile):
            newFile, _ = QFileDialog.getOpenFileName(self, "选择文件", oldFile, "All Files (*)")
        else:
            newFile, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "All Files (*)")
        if newFile:
            editUI.setText(newFile)
    
    def browseOpenPath(self, editUI):
        """浏览打开目录"""
        oldPath = editUI.text()
        if os.path.exists(oldPath):
            newPath = QFileDialog.getExistingDirectory(self, "选择目录", oldPath)
        else:
            newPath = QFileDialog.getExistingDirectory(self, "选择目录")
        if newPath:
            editUI.setText(newPath)
    
    @Slot()
    def startConversion(self):
        """开始执行模型转换"""
        python_path = self.pythonPathEdit.text()
        model_path = self.modelPathEdit.text()
        script_path = self.scriptPathEdit.text()
        save_dir = self.saveDirEdit.text()
        work_dir = self.workDirEdit.text()
        
        # 验证路径有效性
        if not python_path or not os.path.exists(python_path):
            QMessageBox.warning(self, "错误", "请选择有效的Python解释器")
            return
        # 验证路径有效性
        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "错误", "请选择有效的待转换模型文件")
            return
            
        if not script_path or not os.path.exists(script_path):
            QMessageBox.warning(self, "错误", "请选择有效的转换脚本文件")
            return
            
          # 检查是否已有进程在运行
        if self.process and self.process.state() == QProcess.Running:
            QMessageBox.warning(self, "提示", "当前已有转换任务在进行中")
            return
        
        try:
            # 初始化QProcess
            self.process = QProcess(self)
            
            # 显示正在转换
            self.convertBtn.setText("转换中...")
            self.convertBtn.setEnabled(False)
            
            # 连接信号
            self.process.finished.connect(self.processFinished)
            self.process.started.connect(lambda: logger.info("转换进程已启动"))
            
            # 构建命令
            cmd = [
                python_path,  # 使用用户指定的Python解释器
                script_path, 
                "--model_path", model_path, 
                "--save_path", save_dir,
                "--work_dir", work_dir
            ]
            
            logger.info(f"准备执行转换命令: {' '.join(cmd)}")
            
            # 启动进程 (使用工作目录作为当前工作目录)
            execute_dir = work_dir if work_dir else os.path.dirname(script_path)
            if not os.path.exists(execute_dir):
                execute_dir = os.getcwd()  # 使用当前目录作为备用
            
            self.process.start(cmd[0], cmd[1:])  # 第一个参数是程序名，后面是参数
            self.process.setWorkingDirectory(execute_dir)  # 设置工作目录
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动转换进程失败: {str(e)}")
            logger.exception("启动转换进程错误")
            if self.process:
                self.process = None
            self.convertBtn.setText("开始转换")
            self.convertBtn.setEnabled(True)
       
       
        finally:
            self.convertBtn.setText("开始转换")
            self.convertBtn.setEnabled(True)
    @Slot(int, QProcess.ExitStatus)
    def processFinished(self, exitCode):
        """进程结束处理"""
        self.convertBtn.setText("开始转换")
        self.convertBtn.setEnabled(True)
        
        if exitCode == 0:
            QMessageBox.information(self, "成功", "模型转换成功完成")
            logger.info("模型转换成功，进程正常退出")
        else:
            error_msg = "模型转换失败，进程异常退出"
            #if exitStatus == QProcess.CrashExit:
            #    error_msg += "\n进程崩溃退出"
            QMessageBox.critical(self, "失败", error_msg)
            logger.error(f"模型转换失败，退出码: {exitCode}")
        
        self.process = None  # 释放进程对象