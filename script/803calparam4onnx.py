import onnx
import numpy as np
import argparse
import sys
parser = argparse.ArgumentParser(description='计算PyTorch模型的参数量')
parser.add_argument('--model_path', type=str, help='模型文件路径（.pth后缀）')
parser.add_argument('--save_path', type=str, help='保存文件路径')
parser.add_argument('--work_dir', type=str, help='yolo模型结构定义文件目录')
args = parser.parse_args()  # 解析命令行参数
# 使用解析后的参数替代原硬编码路径
model_path = args.model_path
save_path = args.save_path
work_dir = args.work_dir
output=""
sys.path.append(work_dir)
def count_onnx_model_parameters(onnx_model_path):
    model = onnx.load(onnx_model_path)
    total_params = 0
    for node in model.graph.node:
        for input_name in node.input:
            for initializer in model.graph.initializer:
                if input_name == initializer.name:
                    param_shape = initializer.dims
                    param_size = np.prod(param_shape)
                    total_params += param_size
    return total_params


# 使用示例
onnx_model_path = args.model_path
param_count = count_onnx_model_parameters(onnx_model_path)
output+= f"{param_count/1000000:.2f}MB\n"
with open(save_path, 'w', encoding='utf-8') as f:
    f.write(output)
