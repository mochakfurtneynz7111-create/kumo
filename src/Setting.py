import json
from pathlib import Path
from typing import Dict, Optional

SETTING_FILE_PATH = "setting.json"


class Setting:
    def __init__(self):
        # 持久化属性
        self._config_save_path = None
        self._python_interpreter_path = None
        self._chips = None
        self._calparam_script_path = None
        self._inference_script_path = None
        self._inference_workdir = None
        self._simulate_result = None
        self._netron_port = None
        
        # 非持久化属性
        self._dataset_path = None
        self._model_path = None
        self._config_path = None
        self._current_chip = None

    # 使用@property实现智能属性控制
    @property
    def config_save_path(self) -> str:
        return self._config_save_path 

    @config_save_path.setter
    def config_save_path(self, value: str):
        self._config_save_path  = value

    @property
    def python_interpreter_path(self) -> str:
        return self._python_interpreter_path

    @python_interpreter_path.setter
    def python_interpreter_path(self, value: str):
        self._python_interpreter_path = value

    @property
    def chips(self) -> Dict[str, Dict[str, str]]:
        return self._chips
    
    @property
    def current_chip(self) -> Optional[str]:
        return self._current_chip
    
    @current_chip.setter
    def current_chip(self, value: str):
        self._current_chip = value
        
    @property
    def model_path(self) -> Optional[str]:
        return self._model_path
    
    @model_path.setter
    def model_path(self, value: str):
        self._model_path = value
    
    @property
    def config_path(self) -> Optional[str]:
        return self._config_path

    @config_path.setter
    def config_path(self, value: str):
        self._config_path = value
        
    @property
    def dataset_path(self) -> Optional[str]:
        return self._dataset_path
    
    @dataset_path.setter
    def dataset_path(self, value: str):
        self._dataset_path = value
        
    @property
    def inference_script_path(self) -> Optional[str]:
        return self._inference_script_path
    
    @inference_script_path.setter
    def inference_script_path(self, value: str):
        self._inference_script_path = value
        
    @property
    def calparam_script_path(self) -> Optional[str]:
        return self._calparam_script_path
    
    @calparam_script_path.setter
    def calparam_script_path(self, value: str):
        self._calparam_script_path = value

    @property
    def inference_workdir(self) -> str:
        return self._inference_workdir

    @inference_workdir.setter
    def inference_workdir(self, value: str):
        self._inference_workdir = value

    @property
    def simulate_result(self) -> Optional[str]:
        return self._simulate_result
    
    @simulate_result.setter
    def simulate_result(self, value: str):
        self._simulate_result = value
        
    @property
    def netron_port(self) -> Optional[str]:
        return self._netron_port
    
    @netron_port.setter
    def netron_port(self, value: str):
        self._netron_port = value

    def add_chip(self, name: str):
        """动态添加芯片配置"""
        self._chips[name] = {}

    # 配置文件持久化
    def save_to_json(self) -> str:
        json_text = json.dumps({
            "config_save_path": self._config_save_path,
            # "simulate_img_path":self._simulate_img_path,
            "python_interpreter_path": self._python_interpreter_path,
            "dataset_path": self._dataset_path,
            "inference_script_path": self._inference_script_path,
            "calparam_script_path": self._calparam_script_path,
            "inference_workdir": self._inference_workdir,
            "simulate_result": self._simulate_result,
            "netron_port": self._netron_port,
            "chips": self._chips
        }, indent=2, ensure_ascii=False)
        Path(SETTING_FILE_PATH).write_text(json_text, encoding="utf-8")
        # return json_text

    def _update_from_data(self, data: dict) -> None:
        """内部方法：根据字典数据更新实例属性"""
        self._config_save_path = data.get("config_save_path", "")
        self._python_interpreter_path = data.get("python_interpreter_path", "")
        self._dataset_path = data.get("dataset_path", "")
        self._inference_script_path = data.get("inference_script_path", "")
        self._calparam_script_path = data.get("calparam_script_path", "")
        self._inference_workdir = data.get("inference_workdir", "")
        self._simulate_result = data.get("simulate_result", "result/simulate.json")
        self._netron_port = data.get("netron_port", "8080")
        self._chips = data.get("chips", {})

    @classmethod
    def load(cls, data: dict) -> 'Setting':
        instance = cls()
        instance._update_from_data(data)
        return instance

    def reload(self, data: dict) -> None:
        """从 JSON 字符串重新加载配置"""
        self._update_from_data(data)
        

try:
    json_data = json.loads(Path(SETTING_FILE_PATH).read_text(encoding="utf-8"))
except (json.JSONDecodeError, FileNotFoundError):
    # 如果 JSON 解析失败（如文件内容格式错误），文件不存在，创建空 JSON 文件
    setting = Setting.load({})
    setting.save_to_json()
else:
    setting = Setting.load(json_data)