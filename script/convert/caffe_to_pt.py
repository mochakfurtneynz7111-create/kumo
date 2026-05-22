import caffe
import torch
import caffe.proto.caffe_pb2 as caffe_pb2
from google.protobuf import text_format
import numpy as np

def caffe_to_pt(caffe_prototxt_path, caffe_model_path, pt_file_path):
    """
    将 Caffe 模型转换为 PyTorch 模型 (.pt 文件).

    参数:
        caffe_prototxt_path (str):  Caffe prototxt 文件的路径。
        caffe_model_path (str):  Caffe model 文件的路径。
        pt_file_path (str):  保存 PyTorch 模型的路径。
    """
    try:
        # 加载 Caffe 模型定义 (prototxt)
        net = caffe.Net(caffe_prototxt_path, caffe.TEST) # 确保在CPU模式下加载
        net.copy_from(caffe_model_path)

        # 创建一个 PyTorch 模型字典来存储权重
        pytorch_model = {}

        # 遍历 Caffe 的 layers
        for layer_name, layer in net.params.items():
            # 获取权重和偏置 (如果有)
            weights = layer[0].data
            if len(layer) > 1:
                bias = layer[1].data
            else:
                bias = None

            # 将 Caffe 权重转换为 PyTorch 张量, 并保存到字典中
            pytorch_model[layer_name + '.weight'] = torch.from_numpy(weights) # weights转为tensor
            if bias is not None:
                pytorch_model[layer_name + '.bias'] = torch.from_numpy(bias)  # bias转为tensor

        # 保存 PyTorch 模型
        torch.save(pytorch_model, pt_file_path)

        print(f"成功: {caffe_prototxt_path} 和 {caffe_model_path} 转换为 {pt_file_path}")

    except FileNotFoundError:
        print(f"错误: 文件未找到。请检查路径。")
    except Exception as e:
        print(f"发生错误: {e}")

# 示例用法 (需要替换成你自己的文件路径)
if __name__ == '__main__':
    caffe_prototxt = 'deploy.prototxt' # 替换成你的 Caffe prototxt 文件路径
    caffe_model = 'model.caffemodel' # 替换成你的 Caffe model 文件路径
    pt_file = 'model.pt'  # 替换成你想保存的 PyTorch 文件路径
    caffe_to_pt(caffe_prototxt, caffe_model, pt_file)