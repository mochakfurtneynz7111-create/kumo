from src.logic.GetSrcPath import resource_path
from ui.pages.MenuFile_ui import Ui_MenuFile

from src.QTCompat import QWidget, QMenu
from src.QTCompat import QIcon
from src.QTCompat import QPoint
from src.QTCompat import Signal, Slot, Qt
import logging
from src.Setting import setting
logger = logging.getLogger(__name__)


class MenuFile(QWidget, Ui_MenuFile):
    visualizationModelSignal = Signal()
    openSettingSignal = Signal()
    openWeighttransSignal = Signal()
    selectChipSignal = Signal(str)
    compileButtonClicked = Signal(str)
    def __init__(self):
        super(MenuFile, self).__init__()
        self.setupUi(self)
        
        self.editConfigButton.setIcon(QIcon(resource_path(r"src/icon/chip-select.svg")))
        self.compileModelButton.setIcon(QIcon(resource_path(r"src/icon/compile.svg")))
        self.visualizationButton.setIcon(QIcon(resource_path(r"src/icon/tree-node.svg")))
        self.settingButton.setIcon(QIcon(resource_path(r"src/icon/setting.svg")))
        self.weighttransButton.setIcon(QIcon(resource_path(r"src/icon/trans.svg")))
        
        self.editConfigButton.clicked.connect(self.showSelectChip)
        self.setLabelOnClick(self.labelEditConfig, self.showSelectChip)
        
        self.compileModelButton.clicked.connect(lambda : self.compileButtonClicked.emit("compile"))
        self.setLabelOnClick(self.labelCompileModel, lambda : self.compileButtonClicked.emit("compile"))
        
        self.visualizationButton.clicked.connect(lambda : self.visualizationModelSignal.emit())
        self.setLabelOnClick(self.labelVisualization, lambda : self.visualizationModelSignal.emit())
        
        self.settingButton.clicked.connect(lambda: self.openSettingSignal.emit())
        self.setLabelOnClick(self.labeSetting, lambda: self.openSettingSignal.emit())
        
        self.weighttransButton.clicked.connect(lambda: self.openWeighttransSignal.emit())
        self.setLabelOnClick(self.labelWeightTrans, lambda: self.openWeighttransSignal.emit())

        self.chipMenu = QMenu()
        self.chipMenu.setObjectName("ChipMenu")
        
        self.chipMenu.aboutToHide.connect(lambda: self.refresh())
        
        self.updateChipMenu()
        
        # 将布局下所有的按钮改为相同布局
        for child in self.children():
            child.setProperty('class', 'menu-button')
        
        # self.compileButton.clicked.connect(self.compileModel)

        
    @Slot()
    def showSelectChip(self):
        # 获取主窗口的中心点（相对屏幕）
        window_center = self.rect()
        top_center = QPoint(window_center.width() // 2, 0)
        # 获取菜单的尺寸
        menu_size = self.chipMenu.sizeHint()
        
        # 计算菜单左上角的位置（使菜单居中）
        menu_pos = top_center - QPoint(menu_size.width() // 2, 0)
        
        # 显示菜单
        select_action = self.chipMenu.exec_(self.mapToGlobal(menu_pos))
        
        if select_action:
            self.selectChipSignal.emit(select_action.data())
    
    @Slot()
    def refresh(self):
        self.update()

    
    def setLabelOnClick(self, label, slot):
        '''解决点击标签没反应的问题'''
        # label.setCursor(Qt.PointingHandCursor)
        label.mousePressEvent = lambda *args: slot()
    
    
    def updateChipMenu(self):
        self.chipMenu.clear()
        title_action = self.chipMenu.addAction("选择芯片类型")
        title_action.setEnabled(False)  # 设置为不可点击
        for k, v in setting.chips.items():
            action = self.chipMenu.addAction(v.get("name", ""))
            action.setData(k)  # 存储路径信息
        