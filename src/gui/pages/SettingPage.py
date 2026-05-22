from src.QTCompat import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                              QLineEdit, QPushButton, QFileDialog, QGroupBox, QMessageBox,QCheckBox)
from src.QTCompat import Signal, Qt, QEvent, QObject

from src.Setting import setting
import os
import logging

logger = logging.getLogger(__name__)

class SettingPage(QWidget):
    # configChanged = Signal(dict)  # 配置变更信号
    pyEnvChanged = Signal()
    updateConfigSignal = Signal()
    updateChipSignal = Signal()
    def __init__(self):
        super().__init__()
        self.editDict = {}  # 与setting.chips的键对应，修改会先修改该变量，最后确认之后才会同步setting.chips
        # self.deleteChips = []
        self.initUI()
        # self.loadConfig()
        # self.connectSignals()
        
    def initUI(self):
        mainLayout = QVBoxLayout()  # 主布局
        self.mainLayout = mainLayout
        
        mainLayout.setContentsMargins(9, 6, 9, 6)
        mainLayout.setSpacing(3)
        mainLayout.setAlignment(Qt.AlignTop)
        
        # 参数保存路径
        pathLayout = QHBoxLayout()
        pathLayout.setContentsMargins(0, 0, 0, 0)
        pathLablel = QLabel("参数保存路径:")
        pathLablel.setMinimumWidth(80)
        pathLayout.addWidget(pathLablel)
        self.pathEdit = QLineEdit(setting.config_save_path)
        # self.pathEdit.focusOutEvent = lambda e : self.editFocusOut(self.pathEdit, "config_save_path", e)
        pathBrowseBtn = QPushButton("浏览...")
        pathBrowseBtn.clicked.connect(lambda: self.browseOpenPath(self.pathEdit))
        # browse_btn.clicked.connect(self.browse_save_path)
        pathLayout.addWidget(self.pathEdit)
        pathLayout.addWidget(pathBrowseBtn)
        mainLayout.addLayout(pathLayout)
        
        # Python解释器路径
        pythonLayout = QHBoxLayout()
        pythonLayout.setContentsMargins(0, 0, 0, 0)
        pythonLabel = QLabel("Python解释器:")
        pythonLabel.setMinimumWidth(80)
        pythonLayout.addWidget(pythonLabel)
        self.pythonEdit = QLineEdit(setting.python_interpreter_path)
        pythonBowseBtn = QPushButton("浏览...")
        pythonBowseBtn.clicked.connect(lambda *args, edit=self.pythonEdit: self.browseOpenFile(edit))
        # self.pythonEdit.textChanged.connect(self.updateConfig)
        pythonLayout.addWidget(self.pythonEdit)
        pythonLayout.addWidget(pythonBowseBtn)
        mainLayout.addLayout(pythonLayout)
        
        # 数据集路径
        datasetLayout = QHBoxLayout()
        datasetLayout.setContentsMargins(0, 0, 0, 0)
        datasetLabel = QLabel("数据集路径:")
        datasetLabel.setMinimumWidth(80)
        datasetLayout.addWidget(datasetLabel)
        self.datasetEdit = QLineEdit(setting.dataset_path)
        datasetBrowseBtn = QPushButton("浏览...")
        datasetBrowseBtn.clicked.connect(lambda *args, edit=self.datasetEdit: self.browseOpenPath(edit))
        datasetLayout.addWidget(self.datasetEdit)
        datasetLayout.addWidget(datasetBrowseBtn)
        mainLayout.addLayout(datasetLayout)
        
        # netron 端口
        netronPortLayout = QHBoxLayout()
        netronPortLayout.setContentsMargins(0, 0, 0, 0)
        netronPortLabel = QLabel("netron 端口:")
        netronPortLabel.setMinimumWidth(80)
        netronPortLayout.addWidget(netronPortLabel)
        self.netronPortEdit = QLineEdit(setting.netron_port)
        # datasetBrowseBtn = QPushButton("浏览...")
        # datasetBrowseBtn.clicked.connect(lambda *args, edit=self.datasetEdit: self.browseOpenPath(edit))
        netronPortLayout.addWidget(self.netronPortEdit)
        # datasetLayout.addWidget(datasetBrowseBtn)
        mainLayout.addLayout(netronPortLayout)

        # 添加推理路径
        inferenceScriptPathLayout = QHBoxLayout()
        inferenceScriptPathLayout.setContentsMargins(0, 0, 0, 0)
        inferenceScriptPathLabel = QLabel("推理脚本路径:")
        inferenceScriptPathLabel.setMinimumWidth(80)
        inferenceScriptPathLayout.addWidget(inferenceScriptPathLabel)
        self.inferenceScriptPathEdit = QLineEdit(setting.inference_script_path)
        inferenceScriptPathBrowseBtn = QPushButton("浏览...")
        inferenceScriptPathBrowseBtn.clicked.connect(lambda *args, edit=self.inferenceScriptPathEdit: self.browseOpenFile(edit))
        inferenceScriptPathLayout.addWidget(self.inferenceScriptPathEdit)
        inferenceScriptPathLayout.addWidget(inferenceScriptPathBrowseBtn)
        mainLayout.addLayout(inferenceScriptPathLayout)

        # 自定义参数计算脚本路径
        calparamScriptPathLayout = QHBoxLayout()
        calparamScriptPathLayout.setContentsMargins(0, 0, 0, 0)
        calparamScriptPathLabel = QLabel("自定义参数计算脚本路径:")
        calparamScriptPathLabel.setMinimumWidth(80)
        calparamScriptPathLayout.addWidget(calparamScriptPathLabel)
        self.calparamScriptPathEdit = QLineEdit(setting.calparam_script_path)
        calparamScriptPathBrowseBtn = QPushButton("浏览...")
        calparamScriptPathBrowseBtn.clicked.connect(lambda *args, edit=self.calparamScriptPathEdit: self.browseOpenFile(edit))
        calparamScriptPathLayout.addWidget(self.calparamScriptPathEdit)
        calparamScriptPathLayout.addWidget(calparamScriptPathBrowseBtn)
        mainLayout.addLayout(calparamScriptPathLayout)
        
        # 推理工作目录
        inferenceWorkDirLayout = QHBoxLayout()
        inferenceWorkDirLayout.setContentsMargins(0, 0, 0, 0)
        inferenceWorkDirLabel = QLabel("推理工作目录:")
        inferenceWorkDirLabel.setMinimumWidth(80)
        inferenceWorkDirLayout.addWidget(inferenceWorkDirLabel)
        self.inferenceWorkDirEdit = QLineEdit(setting.inference_workdir)
        inferenceWorkDirBrowseBtn = QPushButton("浏览...")
        inferenceWorkDirBrowseBtn.clicked.connect(lambda *args, edit=self.inferenceWorkDirEdit: self.browseOpenPath(edit))
        inferenceWorkDirLayout.addWidget(self.inferenceWorkDirEdit)
        inferenceWorkDirLayout.addWidget(inferenceWorkDirBrowseBtn)
        mainLayout.addLayout(inferenceWorkDirLayout)
        # 芯片配置标签页
        self.initChips()

        # 保存按钮(已经移至外部代码)
        # btnLayout = QHBoxLayout()
        # btnLayout.setAlignment(Qt.AlignRight)
        # saveBtn = QPushButton("保存配置")
        # btnLayout.addWidget(saveBtn)
        # saveBtn.clicked.connect(self.saveConfig)
        
        # mainLayout.addLayout(btnLayout)

        self.setLayout(mainLayout)

    def initChips(self):
        for chipId, chipConfig in setting.chips.items():
            # 创建分组框
            groupBox = QGroupBox(chipConfig.get("name", chipId))
            # groupBox.setCheckable(True)
            
            # 创建垂直布局
            groupBoxLayout = QVBoxLayout()
            groupBoxLayout.setContentsMargins(6, 0, 6, 6)
            groupBoxLayout.setSpacing(3)
            
            # 芯片id与芯片名称
            chipInfoLayout = QHBoxLayout()
            chipInfoLayout.setContentsMargins(0, 0, 0, 0)
            chipInfoLayout.setSpacing(6)
            chipIdLayout = QHBoxLayout()
            chipIdLayout.setContentsMargins(0, 0, 0, 0)
            chipIdLayout.setSpacing(3)
            chipIdLabel = QLabel("芯片ID:")
            chipIdLabel.setMinimumWidth(60)
            chipIdLayout.addWidget(chipIdLabel)
            chipIdEdit = QLineEdit(chipId)
            chipIdEditFilter = EditFilter(chipIdEdit, self.editDict)
            chipIdEdit.installEventFilter(chipIdEditFilter)
            chipIdLayout.addWidget(chipIdEdit)
            chipNameLayout = QHBoxLayout()
            chipNameLayout.setContentsMargins(0, 0, 0, 0)
            chipNameLayout.setSpacing(3)
            chipNameLabel = QLabel("芯片名称:")
            chipNameLabel.setMinimumWidth(60)
            chipNameLayout.addWidget(chipNameLabel)
            chipNameEdit = QLineEdit(chipConfig.get("name", chipId))
            chipNameLayout.addWidget(chipNameEdit)
            deleteBtn = QPushButton("删除")
            deleteBtn.setStyleSheet(
                "QPushButton {"
                "   background-color: rgb(180, 84, 84);"
                "}"
                "QPushButton:hover {"
                "   background-color: rgb(220, 100, 100);"
                "}"
            )
            deleteBtn.clicked.connect(lambda *args, chipIdEdit=chipIdEdit, groupBox=groupBox: self.deleteChip(chipIdEdit, groupBox))
            chipInfoLayout.addLayout(chipIdLayout)
            chipInfoLayout.addLayout(chipNameLayout)
            chipInfoLayout.addWidget(deleteBtn)
            chipInfoLayout.setAlignment(Qt.AlignLeft)
            groupBoxLayout.addLayout(chipInfoLayout)
            
            # 界面ui文件配置
            uiLayout = QHBoxLayout()
            uiLayout.setContentsMargins(0, 0, 0, 0)
            uiLabel = QLabel("参数文件:")
            uiLabel.setMinimumWidth(60)
            uiLayout.addWidget(uiLabel)
            uiEdit = QLineEdit(chipConfig.get("ui_params_path", ""))
            uiLayout.addWidget(uiEdit)
            uiBrowseBtn = QPushButton("浏览...")
            uiBrowseBtn.clicked.connect(lambda *args, edit=uiEdit: self.browseOpenFile(edit))
            uiLayout.addWidget(uiBrowseBtn)
            groupBoxLayout.addLayout(uiLayout)
            
            # 添加预处理命令
            preprocessLayout = QHBoxLayout()
            preprocessLayout.setContentsMargins(0, 0, 0, 0)
            preprocessLabel = QLabel("激活命令:")
            preprocessLabel.setMinimumWidth(60)
            preprocessLayout.addWidget(preprocessLabel)
            preprocessEdit = QLineEdit(chipConfig.get("preprocess", ""))
            preprocessLayout.addWidget(preprocessEdit)
            groupBoxLayout.addLayout(preprocessLayout)
            
            # 添加编译命令配置
            compileLayout = QHBoxLayout()
            compileLayout.setContentsMargins(0, 0, 0, 0)
            compileLabel = QLabel("编译命令:")
            compileLabel.setMinimumWidth(60)
            compileLayout.addWidget(compileLabel)
            compileEdit = QLineEdit(chipConfig.get("compile", ""))

            compileCheckBox = QCheckBox("转换命令行")
            compileCheckBox.setChecked(chipConfig.get("compiletoml2cmd", 0) != 0)

            # compile_edit.textChanged.connect(lambda text: self.updateChipConfig(chipId, "compile", text))
            compileLayout.addWidget(compileEdit)
            compileLayout.addWidget(compileCheckBox)
            # compileBrowseBtn = QPushButton("浏览...")
            # compileBrowseBtn.clicked.connect(lambda *args, edit=compileEdit: self.browseOpenFile(edit))
            # compileLayout.addWidget(compileBrowseBtn)
            groupBoxLayout.addLayout(compileLayout)
            
            
            # 添加编译工作目录配置
            compileWorkDirLayout = QHBoxLayout()
            compileWorkDirLayout.setContentsMargins(0, 0, 0, 0)
            compileWorkDirLabel = QLabel("编译路径:")
            compileWorkDirLabel.setMinimumWidth(60)
            compileWorkDirLayout.addWidget(compileWorkDirLabel)
            compileWorkDirEdit = QLineEdit(chipConfig.get("compile_workdir", ""))
            compileWorkDirLayout.addWidget(compileWorkDirEdit)
            compileWorkDirBrowseBtn = QPushButton("浏览...")
            compileWorkDirBrowseBtn.clicked.connect(lambda *args, edit=compileWorkDirEdit: self.browseOpenPath(edit))
            compileWorkDirLayout.addWidget(compileWorkDirBrowseBtn)
            groupBoxLayout.addLayout(compileWorkDirLayout)
            
                  
            # 添加模拟命令配置
            simulateLayout = QHBoxLayout()
            simulateLayout.setContentsMargins(0, 0, 0, 0)
            simulateLabel = QLabel("模拟命令:")
            simulateLabel.setMinimumWidth(60)
            simulateLayout.addWidget(simulateLabel)
            simulateEdit = QLineEdit(chipConfig.get("simulate", ""))
            # simulate_edit.textChanged.connect(lambda text: self.updateChipConfig(chipId, "simulate", text))
            simulateLayout.addWidget(simulateEdit)
            # simulateBrowseBtn = QPushButton("浏览...")
            # simulateBrowseBtn.clicked.connect(lambda *args, edit=simulateEdit: self.browseOpenFile(edit))
            # simulateLayout.addWidget(simulateBrowseBtn)
            groupBoxLayout.addLayout(simulateLayout)
            
            # 添加模拟工作目录配置
            simulateWorkDirLayout = QHBoxLayout()
            simulateWorkDirLayout.setContentsMargins(0, 0, 0, 0)
            simulateWorkDirLabel = QLabel("模拟目录:")
            simulateWorkDirLabel.setMinimumWidth(60)
            simulateWorkDirLayout.addWidget(simulateWorkDirLabel)
            simulateWorkDirEdit = QLineEdit(chipConfig.get("simulate_workdir", ""))
            simulateWorkDirLayout.addWidget(simulateWorkDirEdit)
            simulateWorkDirBrowseBtn = QPushButton("浏览...")
            simulateWorkDirBrowseBtn.clicked.connect(lambda *args, edit=simulateWorkDirEdit: self.browseOpenPath(edit))
            simulateWorkDirLayout.addWidget(simulateWorkDirBrowseBtn)
            groupBoxLayout.addLayout(simulateWorkDirLayout)
            
            # 设置分组框布局
            groupBox.setLayout(groupBoxLayout)
            
            # 添加到主布局
            self.mainLayout.addWidget(groupBox)
            self.editDict[chipId] = {
                "name": chipNameEdit,
                "ui_params_path": uiEdit,
                "preprocess": preprocessEdit,
                "compile": compileEdit, 
                "compiletoml2cmd": compileCheckBox,
                "compile_workdir": compileWorkDirEdit,
                "simulate": simulateEdit,
                "simulate_workdir": simulateWorkDirEdit
            }
    
    def browseOpenFile(self, editUI):
        """浏览保存路径"""
        oldFile = editUI.text()
        if os.path.exists(oldFile):
            newFile, _ = QFileDialog.getOpenFileName(self, "选择路径", oldFile, "All Files (*);")
        else:
            newFile, _ = QFileDialog.getOpenFileName(self, "选择路径", "", "All Files (*);")
        if newFile and newFile != oldFile:
            editUI.setText(newFile)


    def browseOpenPath(self, editUI):
        """浏览保存路径"""
        oldPath = editUI.text()
        if os.path.exists(oldPath):
            newPath = QFileDialog.getExistingDirectory(self, "选择路径", oldPath)
        else:
            newPath = QFileDialog.getExistingDirectory(self, "选择路径")
        if newPath and newPath != oldPath:
            editUI.setText(newPath)


    def saveConfig(self):
        """保存配置"""
        logger.info("update config")
        
        if setting.config_save_path != self.pathEdit.text():
            setting.config_save_path = self.pathEdit.text()
            self.updateConfigSignal.emit()
            
        if setting.python_interpreter_path != self.pythonEdit.text():
            setting.python_interpreter_path = self.pythonEdit.text()
            self.pyEnvChanged.emit()
            
        setting.dataset_path = self.datasetEdit.text()
        setting.inference_script_path = self.inferenceScriptPathEdit.text()
        setting.calparam_script_path = self.calparamScriptPathEdit.text()
        setting.inference_workdir = self.inferenceWorkDirEdit.text()
        setting.netron_port  = self.netronPortEdit.text()
            
        # 保存芯片配置
        updateChipFlag = False
        for chipId, edits in self.editDict.items():
            if chipId not in setting.chips:    # 新加芯片
                setting.add_chip(chipId)
                updateChipFlag = True
            elif setting.chips[chipId]["name"] != self.editDict[chipId]["name"].text():  # 芯片名称修改
                updateChipFlag = True
            for key, edit in edits.items():
                if isinstance(edit,QLineEdit):
                     setting.chips[chipId][key] = edit.text()
                elif isinstance(edit,QCheckBox):
                      setting.chips[chipId][key] = 1 if edit.isChecked() else 0
            # setting.chips[chipId]['compile'] = edits['compile'].text()
            # setting.chips[chipId]['simulate'] = edits['simulate'].text()
            # setting.chips[chipId]['inference_script_path'] = edits['inferenceScriptPath'].text()

        for chipId in list(setting.chips.keys()):  # 使用 list() 创建副本，删除或者更新芯片
            if chipId not in self.editDict.keys():
                setting.chips.pop(chipId)
                updateChipFlag = True
        
        # for deleteChipId in self.deleteChips:
        #     print('deleteChipId', deleteChipId)
        #     setting.chips.pop(chipId)
        if updateChipFlag:  # 芯片配置有更新
            self.updateChipSignal.emit()
        
        setting.save_to_json()
        QMessageBox.information(self, "保存成功", "配置保存成功！")

    
    def deleteChip(self, chipIdEdit, groupBox):
        """删除芯片"""
        message = QMessageBox.warning(
            self, 
            "删除芯片", 
            "确定要删除该芯片吗？", 
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No  # 默认选中No按钮
        )
        chipId = chipIdEdit.text().strip()
        if message == QMessageBox.Yes:
            self.mainLayout.removeWidget(groupBox)
            groupBox.deleteLater()
            if chipId and chipId in self.editDict:
                self.editDict.pop(chipId)
                # self.deleteChips.append(chipId)

    
    def editFocusOut(self, editUI, attrName, event):
        """失焦后保存数据"""
        print("updateConfig")
        # 调用父类方法确保默认行为
        QLineEdit.focusOutEvent(editUI, event)
        setattr(setting, attrName, editUI.text())

    def addChip(self):
        """添加芯片"""
        # 创建分组框
        groupBox = QGroupBox("new chip")
        # groupBox.setCheckable(True)
        
        # 创建垂直布局
        groupBoxLayout = QVBoxLayout()
        groupBoxLayout.setContentsMargins(6, 0, 6, 6)
        groupBoxLayout.setSpacing(3)
        
        # 芯片id与芯片名称
        chipInfoLayout = QHBoxLayout()
        chipInfoLayout.setContentsMargins(0, 0, 0, 0)
        chipInfoLayout.setSpacing(6)
        chipIdLayout = QHBoxLayout()
        chipIdLayout.setContentsMargins(0, 0, 0, 0)
        chipIdLayout.setSpacing(3)
        chipIdLabel = QLabel("芯片ID:")
        chipIdLabel.setMinimumWidth(60)
        chipIdLayout.addWidget(chipIdLabel)
        chipIdEdit = QLineEdit()
        chipIdEdit.setObjectName("chipIdEdit")
        chipIdLayout.addWidget(chipIdEdit)
        chipNameLayout = QHBoxLayout()
        chipNameLayout.setContentsMargins(0, 0, 0, 0)
        chipNameLayout.setSpacing(3)
        chipNameLabel = QLabel("芯片名称:")
        chipNameLabel.setMinimumWidth(60)
        chipNameLayout.addWidget(chipNameLabel)
        chipNameEdit = QLineEdit()
        chipNameLayout.addWidget(chipNameEdit)
        confirmBtn = QPushButton("确定")
        confirmBtn.setObjectName("confirmBtn")
        confirmBtn.setStyleSheet(
            "QPushButton {"
            "   background-color: rgb(84, 180, 84);"
            "}"
            "QPushButton:hover {"
            "   background-color: rgb(100, 220, 100);"
            "}"
        )
        deleteBtn = QPushButton("删除")
        deleteBtn.setObjectName("deleteBtn")
        deleteBtn.setStyleSheet(
            "QPushButton {"
            "   background-color: rgb(180, 84, 84);"
            "}"
            "QPushButton:hover {"
            "   background-color: rgb(220, 100, 100);"
            "}"
        )
        deleteBtn.clicked.connect(lambda : (
            self.mainLayout.removeWidget(groupBox),
            groupBox.deleteLater()
        ))
        
        chipInfoLayout.addLayout(chipIdLayout)
        chipInfoLayout.addLayout(chipNameLayout)
        chipInfoLayout.addWidget(confirmBtn)
        chipInfoLayout.addWidget(deleteBtn)
        chipInfoLayout.setAlignment(Qt.AlignLeft)
        groupBoxLayout.addLayout(chipInfoLayout)
        
        # 界面ui文件配置
        uiLayout = QHBoxLayout()
        uiLayout.setContentsMargins(0, 0, 0, 0)
        uiLabel = QLabel("参数文件:")
        uiLabel.setMinimumWidth(60)
        uiLayout.addWidget(uiLabel)
        uiEdit = QLineEdit()
        uiLayout.addWidget(uiEdit)
        uiBrowseBtn = QPushButton("浏览...")
        uiBrowseBtn.clicked.connect(lambda *args, edit=uiEdit: self.browseOpenFile(edit))
        uiLayout.addWidget(uiBrowseBtn)
        groupBoxLayout.addLayout(uiLayout)
        
        # 添加预处理命令
        preprocessLayout = QHBoxLayout()
        preprocessLayout.setContentsMargins(0, 0, 0, 0)
        preprocessLabel = QLabel("激活命令:")
        preprocessLabel.setMinimumWidth(60)
        preprocessLayout.addWidget(preprocessLabel)
        preprocessEdit = QLineEdit()
        preprocessLayout.addWidget(preprocessEdit)
        groupBoxLayout.addLayout(preprocessLayout)
            
        
        # 添加编译命令配置
        compileLayout = QHBoxLayout()
        compileLayout.setContentsMargins(0, 0, 0, 0)
        compileLabel = QLabel("编译命令:")
        compileLabel.setMinimumWidth(60)
        compileLayout.addWidget(compileLabel)
        compileEdit = QLineEdit()
        compileCheckBox = QCheckBox("转换命令行")
        compileCheckBox.setChecked(False)
        # compile_edit.textChanged.connect(lambda text: self.updateChipConfig(chipId, "compile", text))
        compileLayout.addWidget(compileEdit)
        compileLayout.addWidget(compileCheckBox)
        # compileBrowseBtn = QPushButton("浏览...")
        # compileBrowseBtn.clicked.connect(lambda *args, edit=compileEdit: self.browseOpenFile(edit))
        # compileLayout.addWidget(compileBrowseBtn)
        groupBoxLayout.addLayout(compileLayout)
        
        # 添加编译工作目录配置
        compileWorkDirLayout = QHBoxLayout()
        compileWorkDirLayout.setContentsMargins(0, 0, 0, 0)
        compileWorkDirLabel = QLabel("编译目录:")
        compileWorkDirLabel.setMinimumWidth(60)
        compileWorkDirLayout.addWidget(compileWorkDirLabel)
        compileWorkDirEdit = QLineEdit()
        compileWorkDirLayout.addWidget(compileWorkDirEdit)
        compileWorkDirBrowseBtn = QPushButton("浏览...")
        compileWorkDirBrowseBtn.clicked.connect(lambda *args, edit=compileWorkDirEdit: self.browseOpenPath(edit))
        compileWorkDirLayout.addWidget(compileWorkDirBrowseBtn)
        groupBoxLayout.addLayout(compileWorkDirLayout)
        
        # 添加模拟命令配置
        simulateLayout = QHBoxLayout()
        simulateLayout.setContentsMargins(0, 0, 0, 0)
        simulateLabel = QLabel("模拟命令:")
        simulateLabel.setMinimumWidth(60)
        simulateLayout.addWidget(simulateLabel)
        simulateEdit = QLineEdit()
        # simulate_edit.textChanged.connect(lambda text: self.updateChipConfig(chipId, "simulate", text))
        simulateLayout.addWidget(simulateEdit)
        # simulateBrowseBtn = QPushButton("浏览...")
        # simulateBrowseBtn.clicked.connect(lambda *args, edit=simulateEdit: self.browseOpenFile(edit))
        # simulateLayout.addWidget(simulateBrowseBtn)
        groupBoxLayout.addLayout(simulateLayout)
        
        # 添加模拟工作目录配置
        simulateWorkDirLayout = QHBoxLayout()
        simulateWorkDirLayout.setContentsMargins(0, 0, 0, 0)
        simulateWorkDirLabel = QLabel("模拟目录:")
        simulateWorkDirLabel.setMinimumWidth(60)
        simulateWorkDirLayout.addWidget(simulateWorkDirLabel)
        simulateWorkDirEdit = QLineEdit()
        simulateWorkDirLayout.addWidget(simulateWorkDirEdit)
        simulateWorkDirBrowseBtn = QPushButton("浏览...")
        simulateWorkDirBrowseBtn.clicked.connect(lambda *args, edit=simulateWorkDirEdit: self.browseOpenPath(edit))
        simulateWorkDirLayout.addWidget(simulateWorkDirBrowseBtn)
        groupBoxLayout.addLayout(simulateWorkDirLayout)
        
        # 设置分组框布局
        groupBox.setLayout(groupBoxLayout)
        self.mainLayout.addWidget(groupBox)

        ui_dict = {
                "name": chipNameEdit,
                "ui_params_path": uiEdit,
                "preprocess": preprocessEdit,
                "compile": compileEdit, 
                "compiletoml2cmd" : compileCheckBox,
                "compile_workdir": compileWorkDirEdit,
                "simulate": simulateEdit,
                "simulate_workdir": simulateWorkDirEdit
            }
        
        confirmBtn.clicked.connect(lambda: self.confirmAddChip(groupBox, ui_dict))

    def confirmAddChip(self, groupBox, ui_dict):
        chipIdEdit = groupBox.findChild(QLineEdit, "chipIdEdit")
        confirmBtn = groupBox.findChild(QPushButton, "confirmBtn")
        deleteBtn = groupBox.findChild(QPushButton, "deleteBtn")
        
        chipId = chipIdEdit.text().strip()
        if not chipId:
            QMessageBox.information(self, "提示", "请输入芯片ID")
            return

        if chipId in setting.chips:
            QMessageBox.information(self, "提示", "ID重复")
            return
        
        confirmBtn.setVisible(False)

        self.editDict[chipId] = ui_dict
        chipIdEdit.setText(chipId)
        deleteBtn.clicked.disconnect()
        deleteBtn.clicked.connect(lambda *args, chipIdEdit=chipIdEdit, groupBox=groupBox: self.deleteChip(chipIdEdit, groupBox))
        chipIdEditFilter = EditFilter(chipIdEdit, self.editDict)
        chipIdEdit.installEventFilter(chipIdEditFilter)
        
def updateChipId(chipId, newChipId, editDict):
    if chipId != newChipId:
        # logger.info(f"updateChipId: {chipId} -> {newChipId}")
        if chipId in editDict:
            editDict[newChipId] = editDict.pop(chipId)
        # if chipId in setting.chips:
        #     setting.chips[newChipId] = setting.chips.pop(chipId)
            # setting.save_to_json()
            


class EditFilter(QObject):
    def __init__(self, lineedit, editDict):
        super().__init__(lineedit)
        self.lineedit = lineedit
        self.editDict = editDict
        self.lineedit.setEnabled(False)
        self.oldText = self.lineedit.text()
        
    def eventFilter(self, obj, event):
        # 处理双击事件
        new_text = self.lineedit.text().strip()  # 去除首尾空格
        if (obj is self.lineedit and 
            event.type() == QEvent.MouseButtonDblClick):
            self.oldText = new_text
            self.lineedit.setEnabled(True)
            self.lineedit.setFocus()
            return True
        
        # 处理失焦事件
        if (obj is self.lineedit and 
            event.type() == QEvent.FocusOut):
            if new_text != self.oldText:
                # if not new_text or new_text in setting.chips.keys():
                if not new_text or new_text in self.editDict.keys():
                    QMessageBox.warning(self.lineedit, "错误", "芯片id重复或id错误")
                    self.lineedit.setText(self.oldText)  # 恢复原值
                    # return True  # 拦截事件
                else:
                    updateChipId(self.oldText, self.lineedit.text(), self.editDict)
            
            self.lineedit.setEnabled(False)
        return super().eventFilter(obj, event)

    # def updateChipConfig(self, chipId, attrName, value):
    #     """更新芯片配置"""
    #     chipConfig = setting.chips.get(chipId, {})
    #     chipConfig[attrName] = value
    #     setting.chips[chipId] = chipConfig
    #     self.configChanged.emit(setting.chips)