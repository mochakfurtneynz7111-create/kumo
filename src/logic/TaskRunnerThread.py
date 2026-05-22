from src.QTCompat import QThread, Signal
import logging
logger = logging.getLogger(__name__)

class TaskRunnerThread(QThread):
    """
    通用线程模板类
    功能：执行任意耗时任务并通过信号返回结果
    """
    finished = Signal(object)  # 任务完成信号（携带结果）
    error = Signal(Exception)  # 错误信号

    def __init__(self, parent, task_func=None, *args, **kwargs):
        """
        :param task_func: 要执行的任务函数
        :param args/kwargs: 任务函数的参数
        """
        super().__init__(parent=parent)
        self.task_func = task_func
        self.args = args
        self.kwargs = kwargs
        self._is_running = False  # 线程状态标志

    def run(self):
        """线程核心逻辑"""
        try:
            self._is_running = True
            result = self.task_func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            logger.error(e)
            self.error.emit(e)
        finally:
            self._is_running = False

    # def stop(self, force=False):
    #     """安全停止线程（支持强制中断）
        
    #     Args:
    #         force: 是否强制终止线程（不推荐常规使用）
    #     """
    #     if not self.isRunning():
    #         return

    #     if force:
    #         # 强制终止模式
    #         self.terminate()  # 立即终止线程
    #         self.wait(2000)  # 最多等待2秒确保终止完成
    #     else:
    #         # 优雅停止模式
    #         self._is_running = False
    #         self.wait()