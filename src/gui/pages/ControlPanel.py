from collections import defaultdict
import json
from ui.pages.ControlPanel_ui import Ui_ControlPanel
from src.QTCompat import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QScrollArea, QPushButton
from src.QTCompat import QWebEngineView
from src.QTCompat import QUrl, Qt, Signal, Slot

from netron import server as netron_server

from src.Logging import logging
from src.Setting import setting
# from src.gui.pages.ConfigDialog import ConfigDialog
from src.gui.pages.SettingPage import SettingPage
from src.gui.pages.WeightTransPage import WeightTransPage
from src.logic.DynamicForm import DynamicFormGenerator

logger = logging.getLogger(__name__)

class ControlPanel(QWidget, Ui_ControlPanel):
    updateConfigSignal = Signal()
    pyEnvChanged = Signal()
    updateChipSignal = Signal()
    def __init__(self):
        super(ControlPanel, self).__init__()
        self.setupUi(self)
        
        self.splitter.setSizes([1000, 1000])
        
        self.configWidget.hide()
        self.dataViewerWidget.hide()
        self.configWidget.tabCloseRequested.connect(self.removeConfigTab)
        self.dataViewerWidget.tabCloseRequested.connect(self.removeDataViewerTab)
        self.browser = None
        
        self.componentMapping= defaultdict(QWidget)  # 用来保存各个类型组件是否存在
        self.componentTypeMapping = defaultdict(str)  # 保存各个组对应的类型(标题)
        
    @Slot()
    def showDataViewerWidget(self, modelPath):
        if not self.dataViewerWidget.isVisible():
            self.dataViewerWidget.show()
        
        if self.dataViewerWidget.count() >= 1:
            logging.info("可视化界面已经存在")
            return

        # 创建一个容器QWidget作为标签页内容
        tab_content = QWidget()
        view_layout = QVBoxLayout(tab_content)  # 将布局设置给容器
        view_layout.setContentsMargins(0, 0, 0, 1)
        view_layout.setSpacing(0)
        
        self.browser = QWebEngineView()
        view_layout.addWidget(self.browser)
        
        # 确保上一个netron服务已停止
        if netron_server.status():
            logger.info("正在停止上一个netron服务...")
            netron_server.stop()
            netron_server.wait()
        
        netron_port = setting.netron_port
        if isinstance(netron_port, str):  # 兼容字符串
            netron_port = int(netron_port)
        try:
            self.address = netron_server.start(modelPath, netron_port, browse=False)
            url_address = "http://{}:{}".format(self.address[0], self.address[1])
            self.browser.load(QUrl(url_address))
        except Exception as e:
            logger.error(e)
        
        self.dataViewerWidget.addTab(tab_content, "Netron")

    @Slot()
    def showChipWidget(self, chipType):
        if not self.configWidget.isVisible():
            self.configWidget.show()
        
        chipTypeStr = f"chip_{chipType}"
        
        if chipTypeStr in self.componentMapping:
            # logger.info(f"{chipTypeStr} 已经存在")
            self.configWidget.setCurrentWidget(self.componentMapping[chipTypeStr])
            return
        
        tabContent = QWidget()
        chipName = setting.chips[chipType].get("name", chipType)
        if chipType:
            config_path = setting.chips[chipType]["ui_params_path"]
            try:
                config = json.load(open(config_path))
                configContent = DynamicFormGenerator(tabContent, config)
                # configContent = ConfigDialog('.')
                # viewLayout = QVBoxLayout(tabContent)  # 将布局设置给容器
                # viewLayout.addWidget(configContent)
            except Exception as e:
                viewLayout = QVBoxLayout(tabContent)  # 将布局设置给容器
                viewLayout.setContentsMargins(0, 0, 0, 0)
                viewLayout.setSpacing(0)
                if isinstance(e, FileNotFoundError):  # 兼容文件不存在
                    label = QLabel(f"{chipName} 配置文件不存在")
                else:
                    label = QLabel(f"{chipName} 配置文件加载失败\n错误信息: {e}")
                label.setAlignment(Qt.AlignCenter)
                viewLayout.addWidget(label)
                logger.error(e)
            else:
                configContent.confirm_signal.connect(lambda : self.updateConfigSignal.emit())
        
        else:
            viewLayout = QVBoxLayout(tabContent)  # 将布局设置给容器
            viewLayout.setContentsMargins(0, 0, 0, 0)
            viewLayout.setSpacing(0)
            label = QLabel(chipTypeStr)
            label.setAlignment(Qt.AlignCenter)
            viewLayout.addWidget(label)
        
        tabIndex = self.configWidget.addTab(tabContent, chipName)
        self.configWidget.setCurrentIndex(tabIndex)
        
        self.componentMapping[chipTypeStr] = tabContent
        self.componentTypeMapping[tabContent] = chipTypeStr
        # logger.info("add chip widget")

    @Slot()
    def showSettingWidget(self):
        if not self.configWidget.isVisible():
            self.configWidget.show()
            
        if "setting" in self.componentMapping: 
            # logger.info("配置页面ui经存在")
            self.configWidget.setCurrentWidget(self.componentMapping["setting"])
            return
        
        tabContent = QWidget()
        layout = QVBoxLayout(tabContent)  # 将布局设置给容器
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 将SettingPage放入独立容器
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)  # 允许内容自适应
        settingPage = SettingPage()
        settingPage.pyEnvChanged.connect(lambda : self.pyEnvChanged.emit())  # 监听Python环境变化
        settingPage.updateConfigSignal.connect(lambda : self.updateConfigSignal.emit())
        settingPage.updateChipSignal.connect(lambda : self.updateChipSignal.emit())
        
        scroll.setWidget(settingPage)  # 保持SettingPage内部布局独立

        # 添加到外部布局
        layout.addWidget(scroll)
        # 保存按钮
        btnLayout = QHBoxLayout()
        btnLayout.setContentsMargins(6, 6, 21, 6)
        btnLayout.setSpacing(6)
        btnLayout.setAlignment(Qt.AlignRight)
        addBtn = QPushButton("添加芯片")
        addBtn.clicked.connect(settingPage.addChip)
        saveBtn = QPushButton("保存配置")
        saveBtn.clicked.connect(settingPage.saveConfig)
        btnLayout.addWidget(addBtn)
        btnLayout.addWidget(saveBtn)
        layout.addLayout(btnLayout)
        layout.setStretch(0, 0)
        
        tabIndex = self.configWidget.addTab(tabContent, "设置")
        self.configWidget.setCurrentIndex(tabIndex)
        
        self.componentMapping["setting"] = tabContent
        self.componentTypeMapping[tabContent] = "setting"
        # logger.info("add setting widget")

    @Slot()
    def showWeighttransWidget(self):
        if not self.configWidget.isVisible():
            self.configWidget.show()
            
        if "weighttrans" in self.componentMapping: 
            # logger.info("转化页面UI已经存在")
            self.configWidget.setCurrentWidget(self.componentMapping["weighttrans"])
            return
        
        tabContent = QWidget()
        layout = QVBoxLayout(tabContent)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 使用WeightTransPage作为权重转换页面
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        weightTransPage = WeightTransPage()
        scroll.setWidget(weightTransPage)

        layout.addWidget(scroll)
        
        # 注意：这里不需要再添加保存按钮，因为WeightTransPage内部已经包含了开始转换按钮
        
        tabIndex = self.configWidget.addTab(tabContent, "权重转换")
        self.configWidget.setCurrentIndex(tabIndex)
        
        self.componentMapping["weighttrans"] = tabContent
        self.componentTypeMapping[tabContent] = "weighttrans"
        logger.info("添加权重转换页面")

    @Slot()
    def showOtherWidget(self, title, widget):
        if not self.configWidget.isVisible():
            self.configWidget.show()
        
        if title in self.componentMapping: 
            # logger.info(f"{title} already exists")
            self.configWidget.setCurrentWidget(self.componentMapping[title])
            return
        
        tabIndex = self.configWidget.addTab(widget, title)
        self.configWidget.setCurrentIndex(tabIndex)
        
        self.componentMapping[title] = widget
        self.componentTypeMapping[widget] = title
        # logger.info("add setting widget")
    
    @Slot()
    def removeConfigTab(self, index):
        widget = self.configWidget.widget(index)
        widgetType = self.componentTypeMapping[widget]
        self.componentMapping.pop(widgetType)
        self.componentTypeMapping.pop(widget)
        self.configWidget.removeTab(index)
        if self.configWidget.count() == 0:
            self.configWidget.hide()
        # logger.info("remove {} widget".format(widgetType))

    @Slot()
    def removeDataViewerTab(self, index):
        widget = self.dataViewerWidget.widget(index)  # 获取控件
        self.dataViewerWidget.removeTab(index)
        self.browser.stop()
        self.browser.setPage(None)
        widget.deleteLater()  # 安全删除
        if self.dataViewerWidget.count() == 0:
            self.dataViewerWidget.hide()
        self.clean_netron_server()
        self.browser = None


    def clean_netron_server(self):
        if self.browser is not None:
            self.browser.page().profile().clearHttpCache()
            self.browser.deleteLater()
        
        if netron_server.status():
            netron_server.stop()
            netron_server.wait()
            
            logger.info("Netron server stopped")