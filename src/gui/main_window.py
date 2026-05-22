import sys
from src.QTCompat import QMainWindow, QToolButton, QMenu, QFileDialog, QWidget, QVBoxLayout, QLabel, QMessageBox
from src.QTCompat import QPoint, Slot, Qt
from src.QTCompat import QIcon
import uuid
import time
from src.logic.GetSrcPath import resource_path
from ui.main_window_ui import Ui_MainWindow

from src.gui.pages import *
from src.logic import *
import datetime
from src.Setting import setting
import logging
import os
import configparser
# import time

# print(setting.python_interpreter_path)
# print(setting.chips)
# print(setting.config_save_path)

logger = logging.getLogger(__name__)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        # self.workspace = QDir.homePath()   # 自动适配系统
        self.modelPath = None
        self.tabIcons = (
            {
                "normal": QIcon(resource_path(r"src/icon/function-white.svg")),
                "active": QIcon(resource_path(r"src/icon/function-blue.svg"))
            },
            {
                "normal": QIcon(resource_path(r"src/icon/Chip-white.svg")),
                "active": QIcon(resource_path(r"src/icon/Chip-blue.svg"))
            }
        )

        # 获取python解释器列表
        # self.interpreters = []
        # getPyEnvThread = TaskRunnerThread(self, getPythonVersionList)
        # getPyEnvThread.finished.connect(self.updatePyEnv)
        # getPyEnvThread.start()
        
        self.configfiles = []
        self.updateConfigFileThread()

        
        # 设置页面分隔比例，数值要大才能按照比例
        self.splitter_h.setSizes([240, 400])
        self.splitter_h.setStretchFactor(0, 0)
        self.splitter_h.setStretchFactor(1, 1)
        
        self.splitter_left_v.setSizes([0,2000,2000])
        self.splitter_left_v.setStretchFactor(0, 0)
        self.splitter_left_v.setStretchFactor(1, 1)
        self.splitter_left_v.setStretchFactor(2, 1)
        
        self.splitter_right_v.setSizes([400, 200])
        self.splitter_right_v.setStretchFactor(0, 1)
        self.splitter_right_v.setStretchFactor(1, 0)
        
        # 初始化各个组件
        self._initWidgets()
        
        self.menuFile.selectChipSignal.connect(lambda type : self.controlPanel.showChipWidget(type))
        self.menuFile.openSettingSignal.connect(self.controlPanel.showSettingWidget)
        self.menuFile.openWeighttransSignal.connect(self.controlPanel.showWeighttransWidget)
        self.controlPanel.updateConfigSignal.connect(self.updateConfigFileThread)
        # self.controlPanel.pyEnvChanged.connect(lambda interpreters=self.interpreters : self.updatePyEnv(interpreters))
        self.controlPanel.updateChipSignal.connect(self.menuFile.updateChipMenu)
        self.controlPanel.updateChipSignal.connect(self.updateChipMenu)
        
        # ---------------------------------------------- 测试 ----------------------------------------------
        test_widget = QWidget()  # 在对应的地方写好自己的界面，传到下面的页面中
        test_layout = QVBoxLayout()
        test_layout.addWidget(QLabel("测试"))
        test_layout.setAlignment(Qt.AlignCenter)
        test_widget.setLayout(test_layout)
        self.menuChip.testSignal.connect(lambda : self.controlPanel.showOtherWidget("测试", test_widget))
        
        
    def _initWidgets(self):
        self._initMenu()
        self._initStatusBar()
        # self._initWorkspaceTreeView()
        self._initLoadModelWidget()
        self._initControlPanelWidget()
        self._initSimAndSearchWidget()
        self._initProcessOutput()
    
    
    def _initMenu(self):
        # 初始化菜单栏
        self.menuFile = MenuFile()
        self.menuChip = MenuChip()
        self.menuFile.visualizationModelSignal.connect(lambda : self.controlPanel.showDataViewerWidget(setting.model_path))
        self.menuTabWidget.addTab(self.menuFile, self.tabIcons[0]["active"], "功能")
        # self.menuTabWidget.addTab(self.menuChip, self.tabIcons[1]["normal"], "芯片")
        self.menuTabWidget.setCurrentIndex(0)
        # 记录菜单当前选中索引
        self.lastSelectedIndex = 0
        self.menuTabWidget.currentChanged.connect(self.changeTabWidget)
        

    # 初始化状态栏
    def _initStatusBar(self):
        # 初始化芯片类型
        self.chipButton = QToolButton()
        self.chipButton.setPopupMode(QToolButton.InstantPopup)
        self.statusBar.addPermanentWidget(self.chipButton)
        
        self.chipButton.setText("请选择芯片类型")
            
        self.chipMenu = QMenu()
        self.chipMenu.setObjectName("ChipMenu")
        self.updateChipMenu()
        
        self.chipButton.clicked.connect(self.showSelectChip)
        
        # 初始化配置文件列表
        self.configButton = QToolButton()
        self.configButton.setPopupMode(QToolButton.InstantPopup)
        self.statusBar.addPermanentWidget(self.configButton)
        self.configButton.setText("请选择配置文件")
        self.configMenu  = QMenu()
        
        self.updateConfigFile()
        self.configButton.clicked.connect(self.showConfigFile)
        
        
        # 初始化python解释器
        #self.pyEnvButton = QToolButton()
        #self.pyEnvButton.setPopupMode(QToolButton.InstantPopup)
        #self.statusBar.addPermanentWidget(self.pyEnvButton)
        # self.updatePyEnv()
        # pyEnvCustomAction.setData("custom")
        #self.pyEnvButton.clicked.connect(self.showEnvMenu)
        
        
    def _initWorkspaceTreeView(self):
        # 初始化工作目录
        pass
        # self.workspaceTreeView = WorkSpaceView(workspace=self.workspace)
        # self.verticalLayout_2.addWidget(self.workspaceTreeView)


    def _initLoadModelWidget(self):
        self.modelLoadWidget = ModelLoad()
        # self.showModelLoadWidget.setParent(self.widget_left_middle)
        self.verticalLayout_3.addWidget(self.modelLoadWidget)
        self.modelLoadWidget.modelPathChanged.connect(lambda path : self.modelLoadWidget.changModelPath(self, path))
        self.modelLoadWidget.modelInferenceStart.connect(lambda : self.modelInference())
        
    def modelInference(self):
        if not setting.python_interpreter_path:
            QMessageBox.warning(self, "错误", "用于运行推理的环境路径未设置！")
            return
        if not setting.model_path:
            QMessageBox.warning(self, "错误", "模型目录未设置！")
            return
        if not setting.inference_script_path:
            QMessageBox.warning(self, "错误", "用于运行推理的脚本路径未设置！")
            return
        if not setting.dataset_path:
            QMessageBox.warning(self, "错误", "数据集路径未设置！")
            return
        else:
            self.modelLoadWidget.current_metrics = {"memory": None, "runtime": None,"model": None, "time": None}
            self.modelLoadWidget.current_metrics["model"]= os.path.basename(setting.model_path)
            os.makedirs("tmp", exist_ok=True)
            tmp_path = os.path.abspath("tmp/temp.json")
            with open(tmp_path, 'w') as f:
                pass  # 这里可以添加写入逻辑，如f.write("{}")写入空JSON对象
            if not setting.python_interpreter_path:
                QMessageBox.warning(self, "错误", "用于运行参数计算的环境路径未设置！")
                return
            if not setting.model_path:
                QMessageBox.warning(self, "错误", "模型所在目录未设置！")
                return
            if not setting.inference_script_path:
                QMessageBox.warning(self, "错误", "推理脚本所在目录未设置！")
                return
            if not setting.dataset_path:
                QMessageBox.warning(self, "错误", "数据集所在目录未设置！")
                return
            params=[]
            if sys.platform == "win32":
                params = [setting.python_interpreter_path,setting.inference_script_path,"--model_path",setting.model_path,"--dataset_path",setting.dataset_path,"--save_json",tmp_path,"--work_dir",setting.inference_workdir]
            else:
                paramss = [setting.python_interpreter_path,setting.inference_script_path,"--model_path",setting.model_path,"--dataset_path",setting.dataset_path,"--save_json",tmp_path,"--work_dir",setting.inference_workdir]
                command_str = ' '.join(paramss)
                params=[command_str]
            process = self.processOutput.add_task2("推理", params)
            process.readyReadStandardOutput.connect(lambda *args: self.modelLoadWidget.getmemory(process))
            starttime=time.time()
            process.finished.connect(lambda *args: self.modelLoadWidget.getruntime(starttime))
            process.finished.connect(lambda *args: self.modelLoadWidget.getoutput())
            task_name="推理"
            result_file="result/infer.json"
            process.finished.connect(lambda exitCode: self.modelLoadWidget.on_process_finished(task_name, exitCode,tmp_path,result_file))
            # self.task_pages.append(page)
    
    def _initControlPanelWidget(self):
        self.controlPanel = ControlPanel()
        self.verticalLayout_5.addWidget(self.controlPanel)
        
    
    def _initSimAndSearchWidget(self):
        self.simAndSearchWidget = SimAndSearch()
        self.verticalLayout_4.addWidget(self.simAndSearchWidget)
    
    
    def _initProcessOutput(self):
        self.processOutput = ProcessOutput(self.widget_left_down)
        self.verticalLayout_6.addWidget(self.processOutput)
        self.widget_right_down.setVisible(False)
        # self.simAndSearchWidget.simulateButtonClicked.connect(lambda task_name: self.processOutput.add_task(task_name, 1,setting.dataset_path))
        self.simAndSearchWidget.simulateButtonClicked.connect(self.run_simulate)
        self.menuFile.compileButtonClicked.connect(self.run_compile)
    
    def run_simulate(self, task_name):
        if not setting.current_chip:
            QMessageBox.warning(self, "警告", "请选择芯片类型")
            return
        chiptype = setting.current_chip
        chipx = setting.chips[chiptype]
        cmd = chipx["simulate"]
        working_directory = chipx["simulate_workdir"]
        if not working_directory and not os.path.exists(working_directory):
            working_directory = None

        result_file = setting.simulate_result  # 带文件名的路径
        result_dir = os.path.dirname(result_file)  # 提取目录路径
        os.makedirs(result_dir, exist_ok=True) 
        # print(result_dir)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        uid = uuid.uuid4().hex[:8]  # 生成短 UUID
        tmp_file = os.path.join("tmp", f"tmp_{timestamp}_{uid}.json")
        params = [setting.python_interpreter_path,cmd,"--img",f'"{setting.dataset_path}"',"--output_dir",f'"{os.path.abspath(tmp_file)}"',"--config",f'"{setting.config_path}"']
        # print(params)
        process = self.processOutput.add_task2(task_name,params, working_directory)
        process.finished.connect(lambda exitCode, exitStatus: self.processOutput.on_process_finished(task_name, exitCode, exitStatus,tmp_file,result_file))
        
    def toml2param(self,toml_path):
        #try:
            # 读取TOML文件
            config = configparser.ConfigParser()
            config.read(toml_path)
            args = ""
            # 遍历每个部分和其键值对
            for section, values in config.items():
                for key, value in values.items():
                    # 根据值的类型进行处理
                    if isinstance(value, str):
                        if value.strip()=="":
                            args+=f"--{key} "
                        else:
                            args+=f"--{key}={value} "
                    elif isinstance(value, bool):
                        # 布尔值，转换为小写字符串
                        args+=f"--{key}={str(value).lower()} "
                    elif isinstance(value, (int, float)):
                        # 数字值，直接转换为字符串
                        args+=f"--{key}={str(value)} "
                    elif isinstance(value, list):
                        # 列表值，根据配置文件中的格式处理
                        # 对于类似 "pre_mean = 123.675, 116.28, 103.53" 的格式
                        # TOML会解析为列表，但实际参数可能需要字符串形式
                        if all(isinstance(v, (int, float)) for v in value):
                            # 如果列表元素都是数字，将其转换为逗号分隔的字符串
                            args+=f"--{key}={','.join(map(str, value))} "
                        else:
                            # 其他列表类型，直接使用字符串表示
                            args+=f"--{key}={str(value)} "
                    else:
                        # 其他类型，转换为字符串
                        args+=f"--{key}={str(value)} "
            return args

        #except Exception as e:
        #        print(f"Error reading TOML file: {e}")
        #        return " "
    
    def run_compile(self, task_name):
        if not setting.current_chip:
            QMessageBox.warning(self, "警告", "请选择芯片类型")
            return
        if not setting.config_path:
            QMessageBox.warning(self, "警告", "请选择配置文件")
            return
        chiptype = setting.current_chip
        chipx = setting.chips[chiptype]
        cmd = chipx["compile"]
        temp_working_directory = chipx["compile_workdir"]

        if not temp_working_directory and not os.path.exists(temp_working_directory):
            working_directory = None
        else:
            working_directory = f'{temp_working_directory}'
        
        # process.start("cmd", ["/c",setting.python_interpreter_path +" "+ cmd+" "+"--img"+" " + data_path])
        # print(working_directory)





        
        print(chipx.get("compiletoml2cmd", 0))
        if chipx.get("compiletoml2cmd", 0) != 1:
            params = ["cmd", "/c", cmd, f'{setting.config_path}']
        else:
            pre_command = chipx["preprocess"]
            full_combined_command = f"{pre_command} && {cmd} {self.toml2param(setting.config_path)}"
            if pre_command=="":
                params = [cmd + " " + self.toml2param(setting.config_path)]
            else:
                params = [full_combined_command]
        
        process = self.processOutput.add_task2(task_name, params, working_directory)
        process.finished.connect(lambda : print("finished"))
    
    def changeTabWidget(self, index):
        self.menuTabWidget.setCurrentIndex(index)
        # 如果有上一次选中的标签，重置其图标
        if self.lastSelectedIndex != index:
            self.menuTabWidget.setTabIcon(
                self.lastSelectedIndex,
                self.tabIcons[self.lastSelectedIndex]["normal"]
            )
        
        # 设置新选中标签的图标
        self.menuTabWidget.setTabIcon(index, self.tabIcons[index]["active"])
        self.lastSelectedIndex = index

#    @Slot()
#    def showEnvMenu(self):
#        # 显示菜单并处理选择
#        btnTopRight = self.pyEnvButton.mapToGlobal(self.pyEnvButton.rect().topRight())
#        pos = QPoint(
#            btnTopRight.x() - self.pyEnvMenu.sizeHint().width(),
#            btnTopRight.y() - self.pyEnvMenu.sizeHint().height()
#        )
#        selected_action = self.pyEnvMenu.exec_(pos)
#        
#        if selected_action:
#            if selected_action.data():
#                newEnv = selected_action.text().split("(")[0].strip()
#                self.pyEnvButton.setText(f"{newEnv} ▼")
#                self.statusBar.showMessage(f"已切换到: {newEnv}", 2000)
#                setting.python_interpreter_path = selected_action.text().split("(")[1].split(")")[0]
#                setting.save_to_json()
#    
#    @Slot()
#    def selectPyEnv(self):
#        newFile, _ = QFileDialog.getOpenFileName(self, "选择路径", "", "All Files (*);")
#        if newFile:
#            vesion = getPythonVersion(newFile)
#            self.pyEnvButton.setText(f"{vesion} ▼")
#            self.statusBar.showMessage(f"已切换到: {newFile}", 2000)
#            setting.python_interpreter_path = newFile
#            setting.save_to_json()
#    
#    @Slot()
#    def updatePyEnv(self, interpreters=None):
#        if interpreters is None:
#            interpreters = []
#        
#        self.interpreters = interpreters
#        
#        if setting.python_interpreter_path and os.path.exists(setting.python_interpreter_path):
#            vesion = getPythonVersion(setting.python_interpreter_path)
#            self.pyEnvButton.setText(f"{vesion} ▼")
#            # for interrupter in self.interpreters:
#            #     if interrupter[1] == setting.python_interpreter_path:
#            #         self.pyEnvButton.setText(interrupter[0])
#            #         break
#
#        else:
#            self.pyEnvButton.setText("请选择python解释器")
#            
#        
#        """显示环境选择菜单"""
#        self.pyEnvMenu = QMenu()
#        # 添加检测到的环境
#        for version, path in self.interpreters:
#            action = self.pyEnvMenu.addAction(version + " (" + path + ")")
#            action.setData(path)  # 存储路径信息
#        self.pyEnvMenu.addSeparator()
#        # 添加自定义路径选项
#        self.pyEnvMenu.addAction("选择其他路径...").triggered.connect(self.selectPyEnv)
    
    @Slot()
    def showSelectChip(self):
        # # 获取主窗口的中心点（相对屏幕）
        # window_center = self.rect()
        # top_center = QPoint(window_center.width() // 2, 0)
        # # 获取菜单的尺寸
        # menu_size = self.chipMenu.sizeHint()
        
        # # 计算菜单左上角的位置（使菜单居中）
        # menu_pos = top_center - QPoint(menu_size.width() // 2, 0)
        btnTopLeft = self.chipButton.mapToGlobal(self.chipButton.rect().topLeft())
        # 计算菜单弹出位置（底部对齐）
        pos = QPoint(
            btnTopLeft.x() - self.chipMenu.sizeHint().width() // 2 + self.chipButton.sizeHint().width() // 2,
            btnTopLeft.y() - self.chipMenu.sizeHint().height()  # Y坐标：按钮底部 - 菜单高度
        )
        # 显示菜单
        select_action = self.chipMenu.exec_(pos)
        
        if select_action:
            current_chip = select_action.data()
            # 选择新的芯片重置芯片配置
            if setting.current_chip is not None and setting.current_chip != current_chip:
                setting.config_path = None
                self.configButton.setText("请选择配置文件")
            
            setting.current_chip = select_action.data()
            chipName = setting.chips[setting.current_chip]["name"]
            self.statusBar.showMessage(f"已切换到: {chipName}", 2000)
            
            self.chipButton.setText(chipName)
            # logger.info(f"select : {chipName}")
    
    @Slot()
    def showConfigFile(self):
        btnTopLeft = self.configButton.mapToGlobal(self.configButton.rect().topLeft())
        # 计算菜单弹出位置（底部对齐）
        pos = QPoint(
            btnTopLeft.x() - self.configMenu.sizeHint().width() // 2 + self.configButton.sizeHint().width() // 2,
            btnTopLeft.y() - self.configMenu.sizeHint().height()  # Y坐标：按钮底部 - 菜单高度
        )
        # 显示菜单
        select_action = self.configMenu.exec_(pos)
        
        if select_action:
            newPath = select_action.data()
            if newPath:
                setting.config_path = newPath 
                self.statusBar.showMessage(f"已切换到: {newPath }", 2000)
                
                self.configButton.setText(os.path.basename(newPath))
                # logger.info(f"select : {newPath}")

    @Slot()
    def selectConfigFile(self):
        newFile, _ = QFileDialog.getOpenFileName(self, "选择配置文件", setting.config_save_path, "All Files (*);;TOML Files (*.toml)")
        self.updateConfigFileThread()
        if newFile:
            setting.config_path = newFile
            self.statusBar.showMessage(f"已切换到: {newFile}", 2000)
            self.configButton.setText(os.path.basename(newFile))
            # logger.info(f"select : {newFile}")
    
    @Slot()
    def updateConfigFile(self, configfile=None):
        if configfile is None:
            configfile = []
            
        self.configfiles = configfile
        self.configMenu.clear()

        for file_path in self.configfiles:
            action = self.configMenu.addAction(os.path.basename(file_path))
            action.setData(file_path)
        self.configMenu.addSeparator()
        self.configMenu.addAction("选择其他文件...").triggered.connect(self.selectConfigFile)
        if self.configMenu.isVisible():
            self.configMenu.hide()
            self.showConfigFile()
        # logger.info("update config file")
    
    def updateConfigFileThread(self):
        updateConfigFileThread = TaskRunnerThread(parent=self, task_func=getConfigFileList, directory=setting.config_save_path, num=10)
        updateConfigFileThread.finished.connect(self.updateConfigFile)
        updateConfigFileThread.start()
    
    def closeEvent(self, event):
        self.controlPanel.clean_netron_server()
        event.accept()  # 允许窗口关闭
        
        
    def updateChipMenu(self):
        self.chipMenu.clear()
        title_action = self.chipMenu.addAction("选择芯片类型")
        title_action.setEnabled(False)  # 设置为不可点击
        self.chipMenu.addSeparator()
        
        for k, v in setting.chips.items():
            action = self.chipMenu.addAction(v.get("name", ""))
            action.setData(k)  # 存储路径信息
