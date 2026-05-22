from src.logic.GetSrcPath import resource_path
from ui.pages.MenuChip_ui import Ui_MenuChip

from src.QTCompat import QWidget, QMenu
from src.QTCompat import QIcon
from src.QTCompat import Signal, QPoint
from src.Setting import setting

class MenuChip(QWidget, Ui_MenuChip):
    selectChipSignal = Signal(str)
    testSignal = Signal()
    def __init__(self):
        super(MenuChip, self).__init__()
        self.setupUi(self)
        
        self.selectChipToolButton.setIcon(QIcon(resource_path(r"src/icon/chip-select.svg")))
        self.toolButton_2.setIcon(QIcon(resource_path(r"src/icon/default.svg")))
        self.toolButton_3.setIcon(QIcon(resource_path(r"src/icon/default.svg")))
        
        self.chipMenu = QMenu()
        self.chipMenu.setObjectName("ChipMenu")
        title_action = self.chipMenu.addAction("选择芯片类型")
        title_action.setEnabled(False)  # 设置为不可点击
        # self.ChipMenu.addSeparator()
        # self.chipList = ["复旦微100TAI", "昇腾310", "海思3403", "瑞芯微RK3588"]
        # self.chipList = setting.chips
        
        for k, v in setting.chips.items():
            action = self.chipMenu.addAction(v.get("name", ""))
            action.setData(k)  # 存储路径信息
        
        for child in self.children():
            child.setProperty('class', 'menu-button')

        self.selectChipToolButton.clicked.connect(self.showSelectChip)
        self.toolButton_2.clicked.connect(self.testSignal.emit)
        
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