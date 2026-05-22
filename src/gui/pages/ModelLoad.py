import json
import subprocess
import sys
from ui.pages.ModelLoad_ui import Ui_ModelLoad
from src.QTCompat import QWidget, QFileDialog, QHeaderView, QMessageBox
from src.QTCompat import Signal
from src.QTCompat import QStandardItemModel, QStandardItem
from src.Setting import setting

import os
from src.QTCompat import Signal, QSize, Qt, QProcess, QProcessEnvironment

import time#devfzh引入
import json#devfzh引入
import re#devfzh引入
import datetime
import logging
logger = logging.getLogger(__name__)
class ModelLoad(QWidget, Ui_ModelLoad):
    modelPathChanged = Signal(str)
    modelInferenceStart = Signal()
    modelInferenceRes = Signal()
    def __init__(self):
        super(ModelLoad, self).__init__()
        self.setupUi(self)
        
        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(["参数", "结果"])  # 单列表头
        self.modelInfoTableView.setModel(self.model)
        self.modelInfoTableView.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.modelInfoTableView.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.modelInfoTableView.verticalHeader().setVisible(False)  # 隐藏行号
        
        self.modelLoadButton.clicked.connect(self.loadModel)
        self.ParamCalButton.clicked.connect(lambda *args : self.getparam())
        self.modelInferenceButton.clicked.connect(lambda : self.modelInferenceStart.emit())
        self.modelInferenceButton.clicked.connect(lambda : self.modelInferenceRes.emit())
        self.labelModelSelected.setText("未选择模型路径")
        self.current_metrics = {}
        self.getout=False
        
        
    def loadModel(self):
        """打开文件对话框选择模型路径"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择配置文件",
            "",  # 起始目录为空（默认上次路径）
            "模型文件 (*.pth *.pt *.h5 *.onnx *.caffemodel);;所有文件 (*)"
        )
        
        if path:  # 用户选择了文件（未点击取消）
            self.modelPath = path
            setting.model_path = path
            self.modelLoadButton.setText(f"更换模型路径")  # 显示简短文件名

            self.labelModelSelected.setText("当前模型: " + os.path.basename(path))
            
            self.modelPathChanged.emit(path)#这个帮到main_window里了

    def readparam(self,path): #process作形参，以防提前销毁
        with open(path, 'r',encoding='utf-8') as f:
            output=f.read()
        param_size = "N/A"  # 默认值
        if output:
            param_size = output
        else:
            error_output = self.process.readAllStandardError().data().decode('utf-8')
            QMessageBox.warning(self, "错误", f"{error_output}")
            return

        # 下面加上模型信息
        if self.model.rowCount() > 0:
            self.model.removeRows(0, self.model.rowCount())
        info_pairs = [
            ("参数量", param_size),
        ]
        os.remove(path)
        for name, value in info_pairs:
            self.model.appendRow([
                QStandardItem(name),
                QStandardItem(value)
            ])


    def getparam(self):##################################
        if not setting.python_interpreter_path:
            QMessageBox.warning(self, "错误", "用于运行参数计算的环境路径未设置！")
            return
        if not setting.model_path:
            QMessageBox.warning(self, "错误", "模型所在目录未设置！")
            return
        else:
            if self.model.rowCount() > 0:
                self.model.removeRows(0, self.model.rowCount())
            self.model.appendRow([
                    QStandardItem("参数量"),
                    QStandardItem("计算中")
                ])
            self.process = QProcess()
            # 设置环境变量以确保 python 运行时无缓冲
            env = QProcessEnvironment.systemEnvironment()
            env.insert("PYTHONUNBUFFERED", "1")  # 设置python运行时D无缓冲环境变量
            workdir=setting.inference_workdir
            os.makedirs("tmp", exist_ok=True)
            with open("tmp/temp.json", 'w'):#存放临时的参数数据
                pass   
            tmp_path = os.path.abspath("tmp/temp.json")    
            # 提取配置信息
            python_path = setting.python_interpreter_path
            model_path = setting.model_path
            pthparam_path = os.path.abspath(os.path.join('script', '803calparam4pth.py'))
            onnxparam_path = os.path.abspath(os.path.join('script', '803calparam4onnx.py'))
            h5param_path = os.path.abspath(os.path.join('script', '803calparam4h5.py'))
            caffeparam_path = os.path.abspath(os.path.join('script', '803calparam4caffe.py'))
            if  model_path.endswith('.pth') or model_path.endswith('.pt'):
                calparam_path=pthparam_path
            if  model_path.endswith('.onnx'):
                calparam_path=onnxparam_path
            if  model_path.endswith('.h5'):
                calparam_path=h5param_path
            if  model_path.endswith('.caffemodel'):
                calparam_path=caffeparam_path
            if  self.optionComboBox.currentIndex()!=0:
                calparam_path=setting.calparam_script_path
            #command =  f"{python_path} {calparam_path} --model_path {setting.model_path} --save_path {tmp_path} --work_dir {workdir}"
            interpreter=f"{python_path} "
            script =  [
                calparam_path,
                "--model_path", setting.model_path,
                "--save_path", tmp_path,
                "--work_dir", workdir
            ]
            #params4win = [command]
            #/usr/local/bin/python3.10 script/803calparam4onnx.py --model_path /home/fangziheng/resnet18.onnx --save_path tmp/tempparam.txt --work_dir 123
            # 执行命令并捕获输出
            if sys.platform == "win32":
                #self.process.start("cmd",  ["/c"] + params4win)
                self.process.start(python_path.rstrip(),script)
            else:
                self.process.start(python_path.rstrip(),script)
            self.process.finished.connect(lambda *args: self.readparam(tmp_path))

    
    def changModelPath(self, ui, path):  # ui 为主界面元素实例
        ui.modelPath = path
        setting.model_path = path
        
        
        # 解析输出中的参数量信息（假设格式为 "23.57MB\n"）
        
    def getoutput(self):
        """处理标准输出并更新对应任务的输出框"""
        with open("tmp/temp.json", 'r', encoding='utf-8') as f:
            json_data = json.load(f)  # 直接解析JSON数据
        self.current_metrics.update(json_data) 
        self.getout=True
        self.checkandsave_metrics()


    def getmemory(self,process):
        pid=process.processId()
        if sys.platform == "win32":
            cmd = f'tasklist /FI "PID eq {pid}" | findstr /I "{pid}"'
            result = subprocess.check_output(cmd, shell=True, encoding='cp850')
            parts = result.strip().split()
            memory_str = parts[-2]
            memory_kb = int(memory_str.replace(',', ''))
        else:
            # 使用 ps 命令获取 RSS 内存 (单位: KB)
            cmd = f'ps -p {pid} -o rss='
            result = subprocess.check_output(cmd, shell=True, encoding='utf-8')
            memory_kb = int(result.strip())      
        memory_mb = memory_kb / 1024  
        memoryoccupy = f"{memory_mb:.3f}MB"
        self.current_metrics["memory"]=memoryoccupy
        self.checkandsave_metrics()
    def getruntime(self,starttime):
        endtime=time.time()
        duration = f"{(endtime-starttime):.3f}s"
        self.current_metrics["runtime"]=duration
        self.checkandsave_metrics()
    def checkandsave_metrics(self):
        if(self.current_metrics.get("runtime") is not None and 
           self.current_metrics.get("memory") is not None and
           self.getout):
            self.current_metrics["time"]=datetime.datetime.now().isoformat()
            os.makedirs("result", exist_ok=True)
            try:
                with open("result/infer.json", "r",encoding="utf-8") as f:
                    original_data = json.load(f)
            except:  # 文件为空或格式错误
                    original_data = []  # 初始化为空列表
            if(len(original_data)<10):#10条刷新
                original_data.append(self.current_metrics)
                original_data.sort(key=lambda x: x.get("time",""),reverse=True)
                print("加入",self.current_metrics)
            else:
                original_data[-1]=self.current_metrics
                original_data.sort(key=lambda x: x.get("time",""),reverse=True)
                print("更替",self.current_metrics)
            self.getout=False
            with open("result/infer.json", "w",encoding="utf-8") as f:
                json.dump(original_data,f,ensure_ascii=False,indent=4)
            if self.model.rowCount() > 0:
                self.model.removeRows(0, self.model.rowCount())
            # 准备要显示的指标对
            # 将 current_metrics 中的所有键值对添加到表格中
            for key, value in self.current_metrics.items():
                # 添加到表格
                self.model.appendRow([
                    QStandardItem(key),
                    QStandardItem(value)
                ])
           
    def on_process_finished(self, task_name, exit_code,tmp_file,result_file):
        if exit_code == 0:
            logger.info(f"任务 {task_name} 正常完成")
            if not os.path.exists(tmp_file):
                logger.warning(f"任务 {task_name} 的临时文件 {tmp_file} 不存在")
                return
            try:
                os.remove(tmp_file)
                logger.info(f"已删除临时文件 {tmp_file}")
            except Exception as del_e:
                logger.warning(f"尝试删除临时文件 {tmp_file} 失败：{del_e}")

#############
    