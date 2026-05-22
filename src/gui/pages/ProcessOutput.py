import sys
from src.QTCompat import QWidget, QHBoxLayout, QLabel, QToolButton, QSpacerItem, QSizePolicy, QVBoxLayout, QPlainTextEdit
from src.QTCompat import Signal, QSize, Qt, QProcess, QProcessEnvironment,QMutex
from src.Setting import setting
from ui.pages.ProcessOutput_ui import Ui_ProcessOutput

import logging
import json#devfzh引入
import os

logger = logging.getLogger(__name__)


class ProcessOutput(QWidget, Ui_ProcessOutput):
    finishSignal = Signal()
    modelInferencefinish = Signal()#添加推理完成信号
    write_complete_Signal = Signal()
    def __init__(self):
        super().__init__()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.mutex = QMutex()

        self.splitter.setSizes([120, 160])
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)  # 右侧默认不拉伸
        self.splitter.setChildrenCollapsible(False)
        self.verticalLayout_2.addStretch(1)  # 底部弹性空间
        self.stackedWidget.setCurrentIndex(-1)
        
        # 任务管理
        self.task_items = []      # 存储TaskItem
        self.task_outputs = []    # 存储输出控件
        self.current_index = -1   # 当前选中索引
        # self.thread_pool = QThreadPool()
        # self.thread_pool.setMaxThreadCount(5)
        
    def add_task(self, task_name,type=0,data_path=None):
        """添加新任务"""
        # if data_path:
        #     return
        item = TaskItem(task_name)
        task_id = len(self.task_items)
        item.click_requested.connect(lambda item: self.set_selected(self.task_items.index(item)))
        item.delete_requested.connect(lambda item: self.remove_task(self.task_items.index(item)))

        self.task_items.append(item)
        self.verticalLayout_2.insertWidget(self.verticalLayout_2.count() - 1, item)

        # 创建任务输出页面
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        outputTextEdit = QPlainTextEdit()
        outputTextEdit.setReadOnly(True)
        layout.addWidget(outputTextEdit)

        # 创建 QProcess 实例
        process = QProcess(page)

        # 设置环境变量以确保 python 运行时无缓冲
        env = QProcessEnvironment.systemEnvironment()
        
        env.insert("PYTHONUNBUFFERED", "1")  # 设置python运行时D无缓冲环境变量
        process.setProcessEnvironment(env)
        process.readyReadStandardOutput.connect(lambda: self.on_ready_read_standard_output(process, outputTextEdit))
        process.readyReadStandardError.connect(lambda: self.on_ready_read_standard_error(process, outputTextEdit))
        
        # process.finished.connect(lambda : print("执行完毕"))
        chiptype = setting.current_chip
        print(chiptype)
        if chiptype == None:
            return
        if type==0:
            chipx = setting.chips[chiptype]
            cmd = chipx[task_name]
            logger.info(cmd)
            working_directory = chipx["compile_workdir"]
            process.setWorkingDirectory(working_directory)
            process.start("cmd", ["/c", cmd + " " + setting.config_path])
            print( cmd + " " + setting.config_path)
        if type==1:
            chipx = setting.chips[chiptype]
            cmd = chipx[task_name]
            logger.info(cmd)
            working_directory = chipx["simulate_workdir"]
            process.setWorkingDirectory(working_directory)
            print(data_path)
            # process.start("cmd", ["/c",setting.python_interpreter_path +" "+ cmd+" "+"--img"+" " + data_path])
            command = f"{setting.python_interpreter_path} {cmd} --img {data_path}"
            # print(command)
            process.start("cmd", ["/c", command])
        else:
            outputTextEdit.appendPlainText("⚠️ 无法识别的任务类型")



        # 连接标准输出
        process.readyReadStandardOutput.connect(lambda: self.on_ready_read_standard_output(process, outputTextEdit))

        self.stackedWidget.addWidget(page)

      
        
        process.start("python", ["script/test.py"])
        process.readyReadStandardOutput.connect(lambda: self.on_ready_read_standard_output(process, outputTextEdit))
        process.readyReadStandardError.connect(lambda: self.on_ready_read_standard_error(process, outputTextEdit))
        # self.task_pages.append(page)
        self.stackedWidget.addWidget(page)
        # 默认选中最后添加的任务
        self.set_selected(task_id)

    def add_task2(self, task_name, params=None, workdir=None):
        if params is None:
            return None
        # 创建任务输出页面
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        outputTextEdit = QPlainTextEdit()
        outputTextEdit.setReadOnly(True)
        layout.addWidget(outputTextEdit)

        # 创建 QProcess 实例
        process = QProcess(page)
        
        # 设置环境变量以确保 python 运行时无缓冲
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")  # 设置python运行时D无缓冲环境变量
        process.setProcessEnvironment(env)
        process.readyReadStandardOutput.connect(lambda: self.on_ready_read_standard_output(process, outputTextEdit))
        process.readyReadStandardError.connect(lambda: self.on_ready_read_standard_error(process, outputTextEdit))
        if workdir is not None:
            process.setWorkingDirectory(workdir)
        logger.info(params)
        logger.info(workdir)
        outputTextEdit.appendPlainText(f"Executing command: {' '.join(params)}\n")
        if sys.platform == "win32":
            process.start(params[0].rstrip(), params[1:])
        else:
            program_name = "bash" # 或者 "sh"
            arguments_list = ["-c", params[0]] # params[0] 是那个包含所有命令的长字符串

            # 正确的调用方式
            process.start(program_name, arguments_list)
            
        process.readyReadStandardOutput.connect(lambda: self.on_ready_read_standard_output(process, outputTextEdit))

        item = TaskItem(task_name)
        task_id = len(self.task_items)
        item.click_requested.connect(lambda item: self.set_selected(self.task_items.index(item)))
        item.delete_requested.connect(lambda item: self.remove_task(self.task_items.index(item)))

        self.task_items.append(item)
        self.verticalLayout_2.insertWidget(self.verticalLayout_2.count() - 1, item)
        if len(self.task_items) > 0:
            self.parent().setVisible(True)  # 显示任务栏
            
        self.stackedWidget.addWidget(page)
        self.set_selected(task_id)
        return process

    def on_process_finished(self, task_name, exit_code, exit_status,tmp_file,result_file):
        print("任务完成调试信息:")
        print(f"tmp_file: {os.path.abspath(tmp_file) if tmp_file else 'None'}")
        print(f"result_file: {os.path.abspath(result_file) if result_file else 'None'}")
        print(f"当前工作目录: {os.getcwd()}")
        if exit_status == QProcess.NormalExit and exit_code == 0:
            logger.info(f"任务 {task_name} 正常完成")
            if not os.path.exists(tmp_file):
                logger.warning(f"任务 {task_name} 的临时文件 {tmp_file} 不存在")
                return

            with open(tmp_file, "r", encoding="utf-8") as tf:
                try:
                    data = json.load(tf)
                except json.JSONDecodeError:
                    logger.warning(f"任务 {task_name} 的临时 JSON 无法解析")
                    raise  # 保留异常抛出行为

            # 阻塞式加锁（没有 try finally，但逻辑清晰）
            self.mutex.lock()
            # 手动锁定后，确保 unlock 会在任意异常后调用
            unlocked = False
            try:
                old_data = []
                if os.path.exists(result_file):
                    with open(result_file, "r", encoding="utf-8") as rf:
                        try:
                            old_data = json.load(rf)
                            if not isinstance(old_data, list):
                                old_data = [old_data]
                        except json.JSONDecodeError:
                            logger.warning("旧的 result.json 无法解析，将使用空列表")

                merged_data = old_data + (data if isinstance(data, list) else [data])
                with open(result_file, "w", encoding="utf-8") as wf:
                    json.dump(merged_data, wf, indent=4, ensure_ascii=False)
                    logger.info(f"已将 {tmp_file} 内容合并写入 {result_file}（共 {len(merged_data)} 条）")
                        # 删除临时文件
                    try:
                        os.remove(tmp_file)
                        logger.info(f"已删除临时文件 {tmp_file}")
                    except Exception as del_e:
                        logger.warning(f"尝试删除临时文件 {tmp_file} 失败：{del_e}")
                unlocked = True
            finally:
                if not unlocked:
                    logger.warning("任务异常终止，正在释放写锁")
                self.mutex.unlock()
                self.write_complete_Signal.emit()

            self.finishSignal.emit()
        else:
            logger.warning(f"任务 {task_name} 异常退出(exit code: {exit_code}, status: {exit_status})")


    def on_ready_read_standard_output(self, process, outputUI):
        """处理标准输出并更新对应任务的输出框"""
        # 获取标准输出
        output = process.readAllStandardOutput().data().decode(errors="ignore")
        outputUI.insertPlainText(output)
    
    def on_ready_read_standard_error(self, process, outputUI):
    
        try:
            error_output = process.readAllStandardError().data().decode('utf-8')
        except Exception as e:
            error_output = "[标准错误读取失败]\n"
            # print(f"[标准错误异常] {e}", flush=True)
        outputUI.insertPlainText(f"[stderr] {error_output}")


    def set_selected(self, index):
        """设置选中状态"""
        # 验证索引有效性
        if index < 0 or index >= len(self.task_items) or index == self.current_index:
            return
            
        # 取消之前选中项
        if self.current_index != -1:
            self.task_items[self.current_index].set_selected(False)
        
        # 设置新选中项
        self.current_index = index
        self.task_items[index].set_selected(True)
        self.stackedWidget.setCurrentIndex(index)
        
        # 更新终端显示
        # self.update_terminal()

    def remove_task(self, index):
        item = self.task_items.pop(index)
        page = self.stackedWidget.widget(index)  # 获取对应页面
        process = page.findChild(QProcess)
        if process.state() == QProcess.Running:
            process.kill()
            # while process.state() == QProcess.Running:
            #     process.waitForFinished(100)
        self.stackedWidget.removeWidget(page)
        
        # 4. 处理选中状态
        if self.current_index == index:
            # 删除的是当前选中项
            if self.task_items:  # 还有剩余任务
                new_index = max(0, min(index, len(self.task_items) - 1))
                self.current_index = new_index
                self.task_items[new_index].set_selected(True)
                self.stackedWidget.setCurrentIndex(new_index)
            else:
                self.current_index = -1
                self.stackedWidget.setCurrentIndex(-1)
                
        elif self.current_index > index:
            self.current_index -= 1
        
        self.verticalLayout_2.removeWidget(item)
        item.deleteLater()
        if len(self.task_items) == 0:
            self.parent().setVisible(False)  # 将自身隐藏


class TaskItem(QWidget):
    delete_requested = Signal(QWidget)  # 自定义删除信号
    click_requested = Signal(QWidget)
    
    def __init__(self, task_name):
        super().__init__()
        self.setAttribute(Qt.WA_StyledBackground)  # 启用样式表背景
        self.setAttribute(Qt.WA_Hover)           # 启用悬停事件
        # self.setMouseTracking(True)               # 精确鼠标跟踪
        
        self.setProperty("selected", False)  # QSS属性控制
        layout = QHBoxLayout()
        layout.setContentsMargins(6, 0, 6, 0)
        # 任务标签
        self.label = QLabel(task_name)
        layout.addWidget(self.label, 1)
        
        # 删除按钮
        layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.delete_btn = QToolButton()
        self.delete_btn.setText("X")
        self.delete_btn.setFixedSize(QSize(16, 16))
        self.delete_btn.setVisible(False)
        self.delete_btn.clicked.connect(lambda: self.delete_requested.emit(self))
        layout.addWidget(self.delete_btn)
        
        self.setLayout(layout)
        
    def enterEvent(self, event):
        # 悬停时改变背景颜色并显示删除按钮
        # self.setStyleSheet("background-color: #e0e0e0;")
        self.delete_btn.setVisible(True)
        
    def leaveEvent(self, event):
        # 鼠标离开时恢复默认背景颜色并隐藏删除按钮
        # self.setStyleSheet("background-color: transparent;")
        self.delete_btn.setVisible(False)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.set_selected(True)
            self.click_requested.emit(self)

    def set_selected(self, selected):
        self.setProperty("selected", selected)
        self.style().unpolish(self)  # 强制刷新样式
        self.style().polish(self)