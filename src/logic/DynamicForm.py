import json
from collections import defaultdict
import configparser

from src.QTCompat import (QWidget, QVBoxLayout, QHBoxLayout, QToolButton,
                              QLabel, QLineEdit, QPushButton, QScrollArea,
                              QRadioButton, QGroupBox, QComboBox, QSpinBox, 
                              QCheckBox, QFileDialog, QMessageBox, QSpacerItem,
                              QApplication, QSizePolicy)
from src.QTCompat import Slot, Qt, Signal

from src.Setting import setting

class DynamicFormGenerator(QWidget):
    confirm_signal = Signal()
    """动态生成表单类"""
    def __init__(self, parent, config):
        super().__init__(parent=parent)
        self.config = config
        self.widgets = defaultdict(dict)  # 自动创建嵌套字典
        self._generate_form(parent)
    
    def _generate_form(self, parent):
        """生成表单"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 3)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignTop)

        scrollarea = QScrollArea()
        scrollarea.setWidgetResizable(True)
        
        # 创建内容控件
        content_widget = QWidget()
        content_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        scrollarea_layout = QVBoxLayout(content_widget)
        scrollarea_layout.setContentsMargins(9, 0, 9, 0)
        scrollarea_layout.setSpacing(12)
        
        for item in self.config:
            # print(item["name"])
            groupbox = QGroupBox(f"[{item['name']}]")
            # 垂直方向设为Fixed防止伸缩
            groupbox.setSizePolicy(
                QSizePolicy.Preferred,  # 水平策略
                QSizePolicy.Fixed       # 垂直策略
            )
            groupbox_layout = QVBoxLayout()
            groupbox_layout.setContentsMargins(6, 3, 6, 6)
            current_row_layout = None  # 当前行布局
            current_row_cnt = 0
            for option in item.get("required_params", []):
                is_text_field = option.get("type") == "text"
                # 文本字段或新行开始
                if is_text_field:
                    current_row_layout = groupbox_layout
                else:  # 非text字段，但是行组件数为0，说明需要创建水平布局
                    if current_row_cnt == 0:
                        # print("创建水平布局")
                        current_row_layout = QHBoxLayout()
                        groupbox_layout.addLayout(current_row_layout)
                    # else:
                    #     print("继续添加组件至水平布局")
                    current_row_cnt += 1
                    
                widget = self._create_field(option, current_row_layout)
                self.widgets[item['name']][option["field"]] = widget
                if is_text_field or current_row_cnt >= 2:
                    # print("行内布局填满或者另起一行")
                    current_row_cnt = 0

            groupbox.setLayout(groupbox_layout)
            scrollarea_layout.addWidget(groupbox)
            
            optional_params = item.get("optional_params", [])
            if optional_params:
                # toolbutton = QToolButton("可选")
                # groupbox.addWidget(toolbutton)
                optional_groupbox = QGroupBox("可选参数")
                optional_groupbox.setAlignment(Qt.AlignCenter)
                # 垂直方向设为Fixed防止伸缩
                optional_groupbox.setSizePolicy(
                    QSizePolicy.Preferred,  # 水平策略
                    QSizePolicy.Fixed       # 垂直策略
                )
                optional_groupbox_layout = QVBoxLayout()
                optional_groupbox_layout.setContentsMargins(6, 3, 6, 6)
                
                current_row_layout = None  # 当前行布局
                current_row_cnt = 0
                for option in optional_params:
                    is_text_field = option.get("type") == "text"
                    if is_text_field:
                        current_row_layout = optional_groupbox_layout
                    else:  # 非text字段，但是行组件数为0，说明需要创建水平布局
                        if current_row_cnt == 0:
                            current_row_layout = QHBoxLayout()
                            optional_groupbox_layout.addLayout(current_row_layout)
                        current_row_cnt += 1
                    widget = self._create_field(option, current_row_layout, True)
                    self.widgets[item['name']][option["field"]] = widget
                    if is_text_field or current_row_cnt >= 2:
                        current_row_cnt = 0
                        
                optional_groupbox.setLayout(optional_groupbox_layout)
                groupbox_layout.addWidget(optional_groupbox)

        
        scrollarea.setWidget(content_widget)
        # 添加提交按钮
        btn_layout = QHBoxLayout()
        load_btn = QPushButton("加载")
        load_btn.setFixedHeight(28)
        load_btn.clicked.connect(lambda *args:self._handle_load(checked=True))
        btn_layout.addWidget(load_btn)
        submit_btn = QPushButton("保存")
        submit_btn.setFixedHeight(28)
        submit_btn.clicked.connect(lambda *args:self._handle_submit(checked=True))
        btn_layout.addWidget(submit_btn)
        btn_layout.addSpacerItem(QSpacerItem(16, 0, QSizePolicy.Fixed, QSizePolicy.Preferred))
        btn_layout.setAlignment(Qt.AlignRight)
        layout.addWidget(scrollarea)
        # layout.addWidget(submit_btn)
        layout.addLayout(btn_layout)
        
        parent.setLayout(layout)
    
    def _create_field(self, option, layout, optional = False):
        widget_type = option.get("type", "text")
        widget_layout = QHBoxLayout()
        widget_layout.setContentsMargins(0, 0, 0, 0)
        label_text = option.get("label") if "label" in option else option.get("field", "")
        label = QLabel(label_text)
        label.setMinimumWidth(80)
        widget_layout.addWidget(label)
            
        if widget_type == "checkbox":
            widget = QCheckBox()
            widget.setMinimumWidth(18)
            widget.setChecked(option.get("default", False))
            widget_layout.addWidget(widget)
            widget_layout.setAlignment(Qt.AlignLeft)
        
        elif widget_type== "select":
            widget = QComboBox()
            widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            items = option.get("options", [])
            labels = option.get("option_labels", items)  # 默认直接使用options作为标签
    
            # 统一转为字符串处理（保留原始值）
            items = [item.strip() if isinstance(item, str) else str(item) for item in items]
            labels = [label.strip() if isinstance(label, str) else str(label) for label in labels]

            if optional and "" not in items:
                items.insert(0, "")  # 添加一个空选项
                labels.insert(0, "")  # 同步添加空标签
            
            for value, label in zip(items, labels):
                widget.addItem(label, userData=value)  # 显示label，存储value
            
            # widget.addItems(items)
            # 设置当前值（根据原始值匹配）
            current_value = str(option.get("default", ""))
            index = widget.findData(current_value)  # 通过userData匹配
            if index >= 0:
                widget.setCurrentIndex(index)
            # widget.setCurrentText(str(option.get("default", "")))
            widget_layout.addWidget(widget)
            widget_layout.setAlignment(Qt.AlignLeft)
            
        elif widget_type == "number":
            widget = QSpinBox()
            minVal = option.get("min", "0")
            maxVal = option.get("max", "100")
            widget.setRange(minVal, maxVal)
            widget.setValue(option.get("default", 0))
            widget_layout.addWidget(widget)
            widget_layout.setAlignment(Qt.AlignLeft)
            
        else:
            widget = QLineEdit()
            defult_value = option.get("default", "")
            if defult_value:
                widget.setText(option.get("default", ""))
            widget_layout.addWidget(widget)
        
        layout.addLayout(widget_layout)
        return widget


    @Slot()
    def _handle_submit(self, checked):
        conf = configparser.ConfigParser()
        for item in self.config:
            part_name = item.get("name")
            # conf[part_name] = {}
            param_dict = {}
            all_params = (option for lst in [item.get("required_params", []), item.get("optional_params", [])] for option in lst)
            # print(all_params)
            
            for option in all_params:
                field = option.get("field")
                widget = self.widgets[part_name][field]
                
                if isinstance(widget, QLineEdit):
                    param_dict[field] = widget.text()
                elif isinstance(widget, QComboBox):
                    # param_dict[field] = widget.currentText()
                     param_dict[field] = widget.currentData() if widget.currentData() is not None else widget.currentText()
                elif isinstance(widget, QCheckBox):
                    # param_dict[field] = "true" if widget.isChecked() else "false"
                    if widget.isChecked():
                        param_dict[field] = ""  # 有就添加空字段
                elif isinstance(widget, QSpinBox):
                    param_dict[field] = str(widget.value())
                else:
                    raise Exception("Invalid widget type")
                
            # new_dict = dict(filter(lambda item: item[1] != "false" and item[1], param_dict.items()))
            # print(param_dict)
            new_dict = dict(filter(lambda item: item[1], param_dict.items()))  # 筛选出非空的字段
            if not new_dict:  # 如果没有非默认值，则不保存该部分配置
                continue
            conf[part_name] = new_dict
        
        file_name, _ = QFileDialog.getSaveFileName(self, "保存文件", setting.config_save_path, "TOML 文件 (*.toml);;INI 文件 (*.ini);;所有文件 (*.*)")
            
        if not file_name:
            return
        
        try:
            with open(file_name, 'w') as configfile:
                conf.write(configfile)
            # 7. 弹出保存成功的提示
            
            self.confirm_signal.emit()  # 将文件名发送到信号槽
            QMessageBox.information(self, "保存成功", f"配置文件已保存至 {file_name}")
        except Exception as e:
            # 发生错误时弹出提示
            QMessageBox.critical(self, "保存失败", f"保存配置文件时发生错误：{e}")
    
    
    def _handle_load(self, checked):
        """从配置文件加载参数"""
        file_name, _ = QFileDialog.getOpenFileName(self, "选择配置文件", setting.config_save_path, "所有文件 (*.*);;TOML 文件 (*.toml);;INI 文件 (*.ini)")
        if not file_name:
            return
        conf = configparser.ConfigParser()
        conf.read(file_name)
        for item in self.config:
            part_name = item.get("name")
            all_params = (option for lst in [item.get("required_params", []), item.get("optional_params", [])] for option in lst)
            
            for option in all_params:
                field = option.get("field")
                widget = self.widgets[part_name][field]
                
                if not conf.has_section(part_name) or not field in conf[part_name]:
                    continue
                
                if isinstance(widget, QLineEdit):
                    widget.setText(conf.get(part_name, field, fallback=option.get("default", "")))
                elif isinstance(widget, QComboBox):
                    # widget.setCurrentText(conf.get(part_name, field, fallback=option.get("default", "")))
                    value = conf.get(part_name, field, fallback=option.get("default", ""))
                    index = widget.findData(value)  # 通过值查找索引
                    if index >= 0:
                        widget.setCurrentIndex(index)  # 显示对应标签
                    else:
                        widget.setCurrentText(str(value))  # 回退：直接显示值（如可编辑模式）
                elif isinstance(widget, QCheckBox):
                    # value = conf.get(part_name, field)
                    # isChecked = value.lower().strip() == ''  # 为空的时候，默认为True，其他值为False
                    widget.setChecked(True)  # 有这个字段就为True
                elif isinstance(widget, QSpinBox):
                    widget.setValue(conf.getint(part_name, field, fallback=option.get("default", 0)))
                else:
                    raise Exception("Invalid widget type")
        
# 使用示例
if __name__ == "__main__":
    import sys
    app = QApplication()
    
    # 示例配置文件路径
    config_path = r"ui_params\fudanwei.json"
    config = json.load(open(config_path))
    
    # 生成界面
    window = QWidget()
    generator = DynamicFormGenerator(config)
    generator.generate_form(window)
    
    window.show()
    sys.exit(app.exec())