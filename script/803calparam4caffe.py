import caffe
import numpy as np
import sys

def calculate_caffe_model_size(prototxt_path, caffemodel_path):
    """计算Caffe模型参数占用的内存大小(MB)"""
    # 加载模型
    net = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)
    
    total_params = 0
    
    # 遍历所有层
    for layer_name in net.params:
        # 遍历该层的所有参数（权重、偏置等）
        for param in net.params[layer_name]:
            # 计算参数数量并累加
            total_params += np.prod(param.data.shape)
    
    # 计算字节数（float32占4字节）并转换为MB
    param_bytes = total_params * 4
    param_mb = param_bytes / (1024 ** 2)
    
    # 输出结果（保留两位小数）
    print(f"{param_mb:.2f} MB")
    
    return param_mb

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python caffe_model_size.py <prototxt_path> <caffemodel_path>")
        sys.exit(1)
    
    prototxt_path = sys.argv[1]
    caffemodel_path = sys.argv[2]
    
    calculate_caffe_model_size(prototxt_path, caffemodel_path)