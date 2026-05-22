import json
import csv
from ui.pages.SimAndSearch_ui import Ui_SimAndSearch
from src.QTCompat import QWidget, QMessageBox, QHeaderView, QVBoxLayout, QDialog, QComboBox, QPushButton, QLabel, QHBoxLayout, QGridLayout
from src.QTCompat import Signal, Qt
from src.QTCompat import QStandardItemModel, QStandardItem
from src.QTCompat import QDoubleSpinBox, QFileDialog, QGroupBox, QRadioButton
import math
import os

class SettingsDialog(QDialog):
    """
    设置对话框，包含权重配置和文件导入功能。
    """

    def __init__(self, parent=None, previous_file_path=None, previous_data_source="simulate"):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setMinimumWidth(450)

        main_layout = QVBoxLayout()

        # ===== 数据源选择区域 =====
        data_source_group = QGroupBox("数据源选择")
        data_source_layout = QVBoxLayout()
        
        self.use_simulate_radio = QRadioButton("使用仿真数据 (simulate.json + infer.json)")
        self.use_import_radio = QRadioButton("使用导入文件（不显示仿真数据）")
        
        # 设置默认选择
        if previous_data_source == "simulate":
            self.use_simulate_radio.setChecked(True)
        else:
            self.use_import_radio.setChecked(True)
        
        # 设置互斥（当一个被选中时，另一个自动取消）
        self.use_simulate_radio.toggled.connect(lambda checked: self.use_import_radio.setChecked(not checked) if checked else None)
        self.use_import_radio.toggled.connect(lambda checked: self.use_simulate_radio.setChecked(not checked) if checked else None)
        
        data_source_layout.addWidget(self.use_simulate_radio)
        data_source_layout.addWidget(self.use_import_radio)
        
        data_source_group.setLayout(data_source_layout)
        main_layout.addWidget(data_source_group)

        # ===== 权重配置区域 =====
        weight_group = QGroupBox("权重配置")
        weight_layout = QGridLayout()

        # Acc 权重
        self.acc_label = QLabel("Accuracy 权重:")
        self.acc_spinbox = QDoubleSpinBox()
        self.acc_spinbox.setRange(-1, 1)
        self.acc_spinbox.setSingleStep(0.01)
        self.acc_spinbox.setValue(0)
        weight_layout.addWidget(self.acc_label, 0, 0)
        weight_layout.addWidget(self.acc_spinbox, 0, 1)

        # Runtime 权重
        self.runtime_label = QLabel("Runtime 权重:")
        self.runtime_spinbox = QDoubleSpinBox()
        self.runtime_spinbox.setRange(-1, 1)
        self.runtime_spinbox.setSingleStep(0.01)
        self.runtime_spinbox.setValue(0)
        weight_layout.addWidget(self.runtime_label, 1, 0)
        weight_layout.addWidget(self.runtime_spinbox, 1, 1)

        # Memory 权重
        self.memory_label = QLabel("Memory 权重:")
        self.memory_spinbox = QDoubleSpinBox()
        self.memory_spinbox.setRange(-1, 1)
        self.memory_spinbox.setSingleStep(0.01)
        self.memory_spinbox.setValue(0)
        weight_layout.addWidget(self.memory_label, 2, 0)
        weight_layout.addWidget(self.memory_spinbox, 2, 1)

        weight_group.setLayout(weight_layout)
        main_layout.addWidget(weight_group)

        # ===== 文件导入区域 =====
        file_group = QGroupBox("数据文件导入")
        file_layout = QVBoxLayout()

        # 文件路径显示和选择
        file_select_layout = QHBoxLayout()
        self.file_path_label = QLabel("未选择文件")
        self.file_path_label.setStyleSheet("QLabel { border: 1px solid #ccc; padding: 5px; }")
        
        # 如果有之前的文件路径，显示它
        if previous_file_path:
            self.file_path = previous_file_path
            self.file_path_label.setText(os.path.basename(previous_file_path))
        else:
            self.file_path = None
        
        self.browse_button = QPushButton("浏览...")
        self.browse_button.clicked.connect(self.browse_file)
        
        self.clear_button = QPushButton("清除")
        self.clear_button.clicked.connect(self.clear_file)
        
        file_select_layout.addWidget(QLabel("数据文件:"))
        file_select_layout.addWidget(self.file_path_label, 1)
        file_select_layout.addWidget(self.browse_button)
        file_select_layout.addWidget(self.clear_button)
        
        file_layout.addLayout(file_select_layout)

        # 文件格式说明
        file_info_label = QLabel(
            "支持格式: Excel (.xlsx, .xls), CSV (.csv), JSON (.json)\n\n"
            "数据要求（必需列/字段）:\n"
            "• Accuracy: 准确率值\n"
            "• Runtime: 运行时间值\n"
            "• Memory: 内存占用值\n\n"
            "注: 不需要Config列，系统会自动生成标识"
        )
        file_info_label.setStyleSheet("QLabel { color: #666; font-size: 10px; padding: 5px; }")
        file_layout.addWidget(file_info_label)

        file_group.setLayout(file_layout)
        main_layout.addWidget(file_group)

        # ===== 确认和取消按钮 =====
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("确定")
        self.cancel_button = QPushButton("取消")
        
        button_layout.addStretch()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)

        main_layout.addLayout(button_layout)

        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        self.setLayout(main_layout)

    def browse_file(self):
        """打开文件选择对话框"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择数据文件",
            "",
            "所有支持的格式 (*.csv *.xlsx *.xls *.json);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            self.file_path = file_path
            self.file_path_label.setText(os.path.basename(file_path))
    
    def clear_file(self):
        """清除选择的文件"""
        self.file_path = None
        self.file_path_label.setText("未选择文件")

    def get_weights(self):
        """获取用户配置的权重值"""
        return {
            "acc": self.acc_spinbox.value(),
            "runtime": self.runtime_spinbox.value(),
            "memory": self.memory_spinbox.value()
        }
    
    def set_weights(self, weights):
        """设置初始权重值"""
        self.acc_spinbox.setValue(weights.get("acc", 0))
        self.runtime_spinbox.setValue(weights.get("runtime", 0))
        self.memory_spinbox.setValue(weights.get("memory", 0))

    def get_file_path(self):
        """获取选择的文件路径"""
        return self.file_path
    
    def get_data_source(self):
        """获取用户选择的数据源"""
        if self.use_simulate_radio.isChecked():
            return "simulate"
        else:
            return "import"


class SimAndSearch(QWidget, Ui_SimAndSearch):
    simulateButtonClicked = Signal(str)

    def __init__(self):
        super(SimAndSearch, self).__init__()
        self.setupUi(self)

        self.simulateButton.clicked.connect(self.simulate)
        self.searchStratecyButton.setText("设置")
        self.searchStratecyButton.clicked.connect(self.open_settings_dialog)
        self.searchButton.clicked.connect(self.start_search)
        self.selectButton.setVisible(False)

        self.model = QStandardItemModel()
        # 初始表头（会根据是否有基准动态调整）
        self.model.setHorizontalHeaderLabels(["Config", "Accuracy", "Runtime", "Memory", "Delta Score", "Score"])
        self.minColumnWidths = [120, 100, 100, 100, 100, 100]
        self.tableView.setModel(self.model)
        header = self.tableView.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        self.tableView.verticalHeader().setVisible(False)
        header.setMinimumSectionSize(100)
        for i, width in enumerate(self.minColumnWidths):
            self.tableView.setColumnWidth(i, width)
        header.sectionResized.connect(self.on_column_resized)
        self._is_resizing = False
        
        self.infer_file_path = None
        self.simulate_file_path = None 

        self.base_metrics = None
        self.simulation_results = {}
        self.weights = None
        self.imported_data = {}  # 存储从外部文件导入的数据
        self.imported_file_path = None  # 保存导入的文件路径
        self.data_source = "simulate"  # 默认使用仿真数据，可选 "simulate" 或 "import"

    def simulate(self):
        self.simulateButtonClicked.emit("simulate") 

    def on_column_resized(self, logical_index, old_size, new_size):
        if self._is_resizing:
            return
        tableView = self.tableView
        total_width = sum(tableView.columnWidth(i) for i in range(tableView.model().columnCount()))
        viewport_width = tableView.viewport().width()

        remaining_width = viewport_width - total_width
        following_columns_count = tableView.model().columnCount() - logical_index
        per_width = remaining_width // following_columns_count
        residual_count = remaining_width % following_columns_count
        
        for i in range(logical_index + 1, tableView.model().columnCount()):
            new_widht = per_width + (1 if i - logical_index <= residual_count else 0)
            tableView.setColumnWidth(i, tableView.columnWidth(i) + per_width)
    
    def resizeEvent(self, event):
        self._is_resizing = True
        self.tableView.resizeRowsToContents()
        header = self.tableView.horizontalHeader()
        total_width = sum(
            self.tableView.columnWidth(i) 
            for i in range(self.tableView.model().columnCount())
        )
        viewport_width = self.tableView.viewport().width()
        
        target_width = viewport_width
        
        scale = target_width / total_width
        for i in range(self.tableView.model().columnCount()):
            self.tableView.setColumnWidth(i,
                                        max(int(self.tableView.columnWidth(i) * scale), 
                                            self.minColumnWidths[i])
                                        )
        self._is_resizing = False
        super().resizeEvent(event)

    def open_settings_dialog(self):
        """打开设置对话框"""
        dialog = SettingsDialog(self, self.imported_file_path, self.data_source)
        if self.weights:
            dialog.set_weights(self.weights)
        
        result = dialog.exec_()

        if result == QDialog.Accepted:
            # 获取权重设置
            self.weights = dialog.get_weights()
            
            # 获取数据源选择
            self.data_source = dialog.get_data_source()
            
            # 获取文件路径
            file_path = dialog.get_file_path()
            
            # 如果选择了导入数据源
            if self.data_source == "import":
                if file_path:
                    # 有文件路径，尝试导入
                    if self.load_external_file(file_path):
                        self.imported_file_path = file_path  # 保存文件路径
                        QMessageBox.information(
                            self, 
                            "成功", 
                            f"已设置权重: {self.weights}\n数据源: 导入文件\n已导入文件: {os.path.basename(file_path)}"
                        )
                    else:
                        QMessageBox.warning(
                            self, 
                            "警告", 
                            f"已设置权重: {self.weights}\n数据源: 导入文件\n但文件导入失败！"
                        )
                elif file_path is None and dialog.file_path_label.text() == "未选择文件":
                    # 用户点击了清除按钮
                    self.imported_file_path = None  # 清除保存的路径
                    self.imported_data = {}  # 清空导入的数据
                    
                    # 清空表格显示
                    if self.model.rowCount() > 0:
                        self.model.removeRows(0, self.model.rowCount())
                    if self.model.columnCount() > 0:
                        self.model.removeColumns(0, self.model.columnCount())
                    
                    QMessageBox.information(
                        self, 
                        "提示", 
                        f"已设置权重: {self.weights}\n数据源: 导入文件\n已清除导入的文件和表格数据"
                    )
                elif self.imported_data:
                    # 没有新文件路径，但已有导入数据
                    QMessageBox.information(
                        self, 
                        "提示", 
                        f"已设置权重: {self.weights}\n数据源: 导入文件\n将使用已导入的数据"
                    )
                else:
                    # 选择了导入但没有文件
                    QMessageBox.warning(
                        self, 
                        "警告", 
                        f"已设置权重: {self.weights}\n数据源: 导入文件\n但未选择文件！请在设置中选择要导入的文件。"
                    )
            else:
                # 选择了仿真数据源
                QMessageBox.information(
                    self, 
                    "提示", 
                    f"已设置权重: {self.weights}\n数据源: 仿真数据 (simulate.json)"
                )

    def load_external_file(self, file_path):
        """
        从外部文件加载数据，支持 Excel、CSV、JSON
        """
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            file_name = os.path.basename(file_path)
            self.imported_data = {}
            
            if file_ext == '.csv':
                return self.load_csv_file(file_path, file_name)
            elif file_ext in ['.xlsx', '.xls']:
                return self.load_excel_file(file_path, file_name)
            elif file_ext == '.json':
                return self.load_json_file(file_path, file_name)
            else:
                QMessageBox.critical(self, "错误", f"不支持的文件格式: {file_ext}")
                return False
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"读取文件时出错: {e}")
            return False

    def load_csv_file(self, file_path, file_name):
        """从CSV文件加载数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                csv_reader = csv.DictReader(f)
                
                # 检查必需的列
                required_columns = ['Accuracy', 'Runtime', 'Memory']
                if not all(col in csv_reader.fieldnames for col in required_columns):
                    QMessageBox.critical(
                        self, 
                        "错误", 
                        f"CSV文件缺少必需的列！\n需要: {', '.join(required_columns)}\n找到: {', '.join(csv_reader.fieldnames)}"
                    )
                    return False
                
                # 读取数据，使用行号作为标识
                row_num = 1
                for row in csv_reader:
                    try:
                        accuracy = float(row['Accuracy'])
                        runtime = float(row['Runtime'])
                        memory = float(row['Memory'])
                        
                        # 生成配置名称：文件名_行号
                        config_name = f"{os.path.splitext(file_name)[0]}_row{row_num}"
                        
                        self.imported_data[config_name] = {
                            "Accuracy": accuracy,
                            "Latency": runtime,
                            "Memory Usage": memory,
                        }
                        row_num += 1
                    except ValueError as e:
                        QMessageBox.warning(
                            self, 
                            "警告", 
                            f"跳过第 {row_num} 行无效数据\n错误: {e}"
                        )
                        row_num += 1
                        continue
                
                if not self.imported_data:
                    QMessageBox.critical(self, "错误", "CSV文件中没有有效数据！")
                    return False
                
                return True
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"读取CSV文件时出错: {e}")
            return False

    def load_excel_file(self, file_path, file_name):
        """从Excel文件加载数据"""
        try:
            # 尝试导入 openpyxl
            try:
                import openpyxl
                workbook = openpyxl.load_workbook(file_path, read_only=True)
                sheet = workbook.active
                
                # 读取表头
                headers = [cell.value for cell in sheet[1]]
                
                # 检查必需的列
                required_columns = ['Accuracy', 'Runtime', 'Memory']
                if not all(col in headers for col in required_columns):
                    QMessageBox.critical(
                        self, 
                        "错误", 
                        f"Excel文件缺少必需的列！\n需要: {', '.join(required_columns)}\n找到: {', '.join(headers)}"
                    )
                    return False
                
                # 获取列索引
                acc_idx = headers.index('Accuracy')
                runtime_idx = headers.index('Runtime')
                memory_idx = headers.index('Memory')
                
                # 读取数据（从第2行开始）
                row_num = 1
                for row in sheet.iter_rows(min_row=2, values_only=True):
                    try:
                        accuracy = float(row[acc_idx])
                        runtime = float(row[runtime_idx])
                        memory = float(row[memory_idx])
                        
                        # 生成配置名称
                        config_name = f"{os.path.splitext(file_name)[0]}_row{row_num}"
                        
                        self.imported_data[config_name] = {
                            "Accuracy": accuracy,
                            "Latency": runtime,
                            "Memory Usage": memory,
                        }
                        row_num += 1
                    except (ValueError, TypeError) as e:
                        row_num += 1
                        continue
                
                workbook.close()
                
            except ImportError:
                QMessageBox.critical(
                    self, 
                    "错误", 
                    "未安装 openpyxl 库！\n请使用命令安装: pip install openpyxl -i https://pypi.tuna.tsinghua.edu.cn/simple"
                )
                return False
            
            if not self.imported_data:
                QMessageBox.critical(self, "错误", "Excel文件中没有有效数据！")
                return False
            
            return True
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"读取Excel文件时出错: {e}")
            return False

    def load_json_file(self, file_path, file_name):
        """从JSON文件加载数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if not isinstance(data, list):
                    QMessageBox.critical(self, "错误", "JSON文件应包含一个数组！")
                    return False
                
                # 读取数据
                row_num = 1
                for item in data:
                    try:
                        # 支持多种字段名
                        accuracy = item.get('Accuracy') or item.get('accuracy') or item.get('acc')
                        runtime = item.get('Runtime') or item.get('runtime') or item.get('Latency') or item.get('latency')
                        memory = item.get('Memory') or item.get('memory') or item.get('Memory Usage')
                        
                        if accuracy is None or runtime is None or memory is None:
                            row_num += 1
                            continue
                        
                        accuracy = float(accuracy)
                        runtime = float(runtime)
                        memory = float(memory)
                        
                        # 生成配置名称
                        config_name = f"{os.path.splitext(file_name)[0]}_row{row_num}"
                        
                        self.imported_data[config_name] = {
                            "Accuracy": accuracy,
                            "Latency": runtime,
                            "Memory Usage": memory,
                        }
                        row_num += 1
                    except (ValueError, TypeError) as e:
                        row_num += 1
                        continue
                
                if not self.imported_data:
                    QMessageBox.critical(self, "错误", "JSON文件中没有有效数据！")
                    return False
                
                return True
                
        except json.JSONDecodeError:
            QMessageBox.critical(self, "错误", "JSON文件格式错误！")
            return False
        except Exception as e:
            QMessageBox.critical(self, "错误", f"读取JSON文件时出错: {e}")
            return False
    
    def load_simulate_data(self, num_entries=5):
        """读取 simulate.json 文件"""
        print("load!")
        self.simulation_results = {}
         
        try:
            print("读取setting啦！")
            with open("setting.json", 'r', encoding="utf-8") as f:
                settings = json.load(f)
                self.simulate_file_path = settings.get("simulate_result", "simulate.json")

            with open(self.simulate_file_path, 'r', encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    latest_data = data[-num_entries:] if len(data) >= num_entries else data
                    for i, item in enumerate(latest_data):
                        config_path = item.get("time", f"Entry {i}")
                        simulation_result = {
                            "Accuracy": item.get("acc", 0.0),
                            "Latency": item.get("runtime", 0.0),
                            "Memory Usage": item.get("memory", 0.0),
                        }
                        self.simulation_results[config_path] = simulation_result
                    return True
                else:
                    QMessageBox.critical(self, "错误", "simulate.json 文件内容应为一个 JSON 列表！")
                    return False
        except FileNotFoundError:
            QMessageBox.critical(self, "错误", "找不到 simulate.json 文件！请先开始仿真")
            return False
        except json.JSONDecodeError:
            QMessageBox.critical(self, "错误", "simulate.json 文件格式错误！")
            return False
        except Exception as e:
            QMessageBox.critical(self, "错误", f"读取 simulate.json 时发生未知的错误：{e}")
            return False    

    def load_base_metrics(self):
        """从 infer.json 加载基准指标"""
        try:
            with open("setting.json", 'r', encoding="utf-8") as f:
                settings = json.load(f)
                self.infer_file_path = settings.get("infer_result", "infer.json")

            with open(self.infer_file_path, 'r', encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and data:
                    data.sort(key=lambda x: x.get('time', ''))
                    last_item = data[-1]
                    accuracy = float(last_item.get("accuracy", "0.0").replace("%", "")) / 100.0
                    runtime = float(last_item.get("runtime", "0.0s").replace("s", ""))
                    memory = float(last_item.get("memory", "0.0MB").replace("MB", ""))
                    self.base_metrics = {
                        "Accuracy": accuracy,
                        "Latency": runtime,
                        "Memory Usage": memory,
                    }
                    return True
                else:
                    QMessageBox.critical(self, "错误", f"{self.infer_file_path} 内容为空或不是 JSON 列表！")
                    return False
        except FileNotFoundError:
            return False
        except json.JSONDecodeError:
            QMessageBox.critical(self, "错误", f"{self.infer_file_path} 或 setting.json 格式错误！")
            return False
        except Exception as e:
            QMessageBox.critical(self, "错误", f"读取数据时出错：{e}")
            return False

    def start_search(self):
        """开始搜索"""
        print("开始搜索！")
        
        # 根据用户选择的数据源加载数据
        all_results = {}
        should_use_baseline = False
        
        if self.data_source == "import":
            # 使用导入数据
            if not self.imported_data:
                QMessageBox.warning(self, "警告", "未导入任何数据文件！请在设置中导入数据文件。")
                return
            all_results.update(self.imported_data)
            print(f"使用导入的数据，共 {len(self.imported_data)} 条")
            # 导入数据时不使用基准
            should_use_baseline = False
        else:
            # 使用仿真数据
            if not self.load_simulate_data():
                return
            all_results.update(self.simulation_results)
            print(f"使用仿真数据，共 {len(self.simulation_results)} 条")
            
            # 仿真数据时尝试加载基准
            if not self.base_metrics:
                self.load_base_metrics()
            if self.base_metrics:
                should_use_baseline = True
        
        # 检查权重
        if not self.weights:
            if should_use_baseline:
                QMessageBox.warning(self, "警告", "未选择搜索策略，我们将使用默认权重\n默认权重 acc: 0.0, runtime: 0.0, memory: 0.0！")
                self.weights = {"acc": 0.0, "runtime": 0.0, "memory": 0.0}
            else:
                QMessageBox.information(self, "提示", "将直接对数据排序\n使用均衡权重 acc: 0.33, runtime: 0.33, memory: 0.34")
                self.weights = {"acc": 0.33, "runtime": 0.33, "memory": 0.34}
        
        # 计算并显示结果
        if should_use_baseline:
            # 有基准：显示所有列（包括基准行）
            scored_results = self.calculate_delta_scores(all_results, self.base_metrics, self.weights)
            sorted_results = sorted(scored_results.items(), key=lambda item: abs(item[1]["delta_score"]))
            self.update_table_with_sorted_results(sorted_results, has_baseline=True)
        else:
            # 无基准：不显示 Delta Score 列和基准行
            scored_results = self.calculate_absolute_scores(all_results, self.weights)
            sorted_results = sorted(scored_results.items(), key=lambda item: item[1]["score"], reverse=True)
            self.update_table_with_sorted_results(sorted_results, has_baseline=False)

    def calculate_absolute_scores(self, results, weights):
        """在没有基准指标时，直接基于结果计算绝对分数"""
        scored_results = {}
        
        if not results:
            return scored_results
        
        acc_values = [r["Accuracy"] for r in results.values()]
        runtime_values = [r["Latency"] for r in results.values()]
        memory_values = [r["Memory Usage"] for r in results.values()]
        
        max_acc = max(acc_values) if acc_values else 1
        min_acc = min(acc_values) if acc_values else 0
        max_runtime = max(runtime_values) if runtime_values else 1
        min_runtime = min(runtime_values) if runtime_values else 0
        max_memory = max(memory_values) if memory_values else 1
        min_memory = min(memory_values) if memory_values else 0
        
        for config_path, result in results.items():
            # 归一化
            norm_acc = (result["Accuracy"] - min_acc) / (max_acc - min_acc) if max_acc != min_acc else 1.0
            norm_runtime = 1 - (result["Latency"] - min_runtime) / (max_runtime - min_runtime) if max_runtime != min_runtime else 1.0
            norm_memory = 1 - (result["Memory Usage"] - min_memory) / (max_memory - min_memory) if max_memory != min_memory else 1.0
            
            # 计算加权分数
            weighted_score = (
                abs(weights["acc"]) * norm_acc +
                abs(weights["runtime"]) * norm_runtime +
                abs(weights["memory"]) * norm_memory
            )
            
            weight_sum = abs(weights["acc"]) + abs(weights["runtime"]) + abs(weights["memory"])
            if weight_sum > 0:
                weighted_score = weighted_score / weight_sum
            
            score = weighted_score * 100
            
            scored_results[config_path] = {
                "Accuracy": result["Accuracy"],
                "Latency": result["Latency"],
                "Memory Usage": result["Memory Usage"],
                "delta_score": 0,
                "score": score,
            }
        
        return scored_results

    def calculate_delta_scores(self, results, base_metrics, weights):
        """计算每个配置与基准指标的差距"""
        scored_results = {}
        for config_path, result in results.items():
            delta_acc = (base_metrics["Accuracy"] - result["Accuracy"]) / base_metrics["Accuracy"] if base_metrics["Accuracy"] != 0 else 0
            delta_runtime = (base_metrics["Latency"] - result["Latency"]) / base_metrics["Latency"] if base_metrics["Latency"] != 0 else 0
            delta_memory = (base_metrics["Memory Usage"] - result["Memory Usage"]) / base_metrics["Memory Usage"] if base_metrics["Memory Usage"] != 0 else 0

            delta_score = (
                    weights["acc"] * delta_acc +
                    weights["runtime"] * delta_runtime +
                    weights["memory"] * delta_memory
            )
            
            abs_delta = abs(delta_score)
            if abs_delta == 0:
                score = 100
            else:
                k = 10
                score = 100 * (1 / (1 + k * abs_delta))
            score = max(0, min(100, score))

            scored_results[config_path] = {
                "Accuracy": result["Accuracy"],
                "Latency": result["Latency"],
                "Memory Usage": result["Memory Usage"],
                "delta_score": delta_score,
                "score": score,
            }
        return scored_results

    def update_table_with_sorted_results(self, sorted_results, has_baseline=True):
        """使用排序后的结果更新表格"""
        # 清空现有数据
        if self.model.rowCount() > 0:
            self.model.removeRows(0, self.model.rowCount())
        
        # 清空现有列
        if self.model.columnCount() > 0:
            self.model.removeColumns(0, self.model.columnCount())

        # 根据是否有基准调整表头
        if has_baseline:
            headers = ["Config", "Accuracy", "Runtime", "Memory", "Delta Score", "Score"]
            self.minColumnWidths = [120, 100, 100, 100, 100, 100]
        else:
            headers = ["Config", "Accuracy", "Runtime", "Memory", "Score"]
            self.minColumnWidths = [150, 120, 120, 120, 120]
        
        self.model.setHorizontalHeaderLabels(headers)
        
        # 重新设置列宽
        for i, width in enumerate(self.minColumnWidths):
            if i < len(headers):
                self.tableView.setColumnWidth(i, width)

        row_index = 0
        
        # 只在有基准指标时才添加基准行
        if has_baseline and self.base_metrics:
            config_item = QStandardItem("infer.json (基准)")
            acc_item = QStandardItem(f"{self.base_metrics['Accuracy']:.4f}")
            runtime_item = QStandardItem(f"{self.base_metrics['Latency']:.4f}")
            memory_item = QStandardItem(f"{self.base_metrics['Memory Usage']:.4f}")
            delta_score_item = QStandardItem("0.0000")
            score_item = QStandardItem("100.00")
            
            self.model.setItem(0, 0, config_item)
            self.model.setItem(0, 1, acc_item)
            self.model.setItem(0, 2, runtime_item)
            self.model.setItem(0, 3, memory_item)
            self.model.setItem(0, 4, delta_score_item)
            self.model.setItem(0, 5, score_item)
            
            row_index = 1

        # 添加所有数据
        for config_path, result in sorted_results:
            acc_str = f"{result['Accuracy']:.4f}"
            runtime_str = f"{result['Latency']:.4f}"
            memory_str = f"{result['Memory Usage']:.4f}"
            score_str = f"{result.get('score', 0):.2f}"

            config_item = QStandardItem(config_path)
            acc_item = QStandardItem(acc_str)
            runtime_item = QStandardItem(runtime_str)
            memory_item = QStandardItem(memory_str)
            score_item = QStandardItem(score_str)

            col_index = 0
            self.model.setItem(row_index, col_index, config_item)
            col_index += 1
            self.model.setItem(row_index, col_index, acc_item)
            col_index += 1
            self.model.setItem(row_index, col_index, runtime_item)
            col_index += 1
            self.model.setItem(row_index, col_index, memory_item)
            col_index += 1
            
            # 只在有基准时添加Delta Score列
            if has_baseline:
                delta_score_str = f"{result['delta_score']:.4f}"
                delta_score_item = QStandardItem(delta_score_str)
                self.model.setItem(row_index, col_index, delta_score_item)
                col_index += 1
            
            self.model.setItem(row_index, col_index, score_item)

            row_index += 1

    def update_table_view(self):
        """根据数据更新表格视图"""
        if not self.simulation_results and not self.imported_data:
            return
        
        # 根据数据源选择
        if self.data_source == "import":
            all_results = self.imported_data
            should_use_baseline = False
        else:
            all_results = self.simulation_results
            should_use_baseline = bool(self.base_metrics)
        
        if should_use_baseline:
            scored_results = self.calculate_delta_scores(all_results, self.base_metrics, self.weights)
            sorted_results = sorted(scored_results.items(), key=lambda item: abs(item[1]["delta_score"]))
            self.update_table_with_sorted_results(sorted_results, has_baseline=True)
        else:
            scored_results = self.calculate_absolute_scores(all_results, self.weights)
            sorted_results = sorted(scored_results.items(), key=lambda item: item[1]["score"], reverse=True)
            self.update_table_with_sorted_results(sorted_results, has_baseline=False)