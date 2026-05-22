import sys
import torch
import argparse
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
if model_path.endswith('.pth') or model_path.endswith('.pt'):
    model_dict = torch.load(model_path, map_location=torch.device('cpu'))
    # 打印模型文件的键
    params_dict = model_dict
    total_params = 0
    for key,param_tensor in params_dict.items():
        if isinstance(param_tensor, torch.Tensor):
        # 将当前参数的元素数（即参数大小）加到总和中
            total_params += param_tensor.numel()
        else:
            print(f"跳过非张量参数: {key}(类型: {type(param_tensor)})")
    if 'model' in model_dict and hasattr(model_dict['model'], 'parameters'):
        model = model_dict['model']
        total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params/1000000:.2f}MB")
    output+= f"{total_params/1000000:.2f}MB\n"
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(output)
else:
    print(f"文件 {model_path} 不是.pth 后缀，不执行计算参数量的操作。")
    