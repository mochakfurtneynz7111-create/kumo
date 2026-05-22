import onnx
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import argparse
parser = argparse.ArgumentParser(description='计算PyTorch模型的参数量')
parser.add_argument('--model_path', type=str, help='模型文件路径（.pth后缀）')
parser.add_argument('--save_path', type=str, help='保存文件路径')
args = parser.parse_args()  # 解析命令行参数
model=load_model(args.model_path)
save_path = args.save_path
output=""
# 使用示例

total_params = model.count_params()
output+= f"模型参数量: {total_params/1000000:.2f}MB\n"
with open(save_path, 'w', encoding='utf-8') as f:
    f.write(output)