from src.QTCompat import QFileSystemModel,QWidget
from ui.pages.WorkSpaceView_ui import Ui_workSpaceView


class WorkSpaceView(QWidget, Ui_workSpaceView):
    def __init__(self, workspace=None):
        super(WorkSpaceView, self).__init__()
        self.setupUi(self)
        self.workspace = workspace
        self.workSpaceTreeView.setHeaderHidden(True)  # 隐藏整个表头栏
        self.setFileSystemModel()
    
    def setFileSystemModel(self):
        if self.workspace is None:
            return
        # 创建模型并设置根路径
        model = QFileSystemModel()
        model.setRootPath(self.workspace)  # 设置监控的根目录
        model.setIconProvider(None)  # 关键：禁用图标显示
        # 绑定模型到树视图
        self.workSpaceTreeView.setModel(model)
        self.workSpaceTreeView.setRootIndex(model.index(self.workspace))
        
        # 隐藏无关列（仅保留文件名）
        for col in range(1, model.columnCount()):
            self.workSpaceTreeView.hideColumn(col)