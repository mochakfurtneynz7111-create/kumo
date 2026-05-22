import onnx
from onnx2pytorch import ConvertModel
import torch

def onnx_to_pt(onnx_file_path, pt_file_path):
    """
    将 ONNX 模型转换为 PyTorch 模型 (.pt 文件).

    参数:
        onnx_file_path (str):  ONNX 模型的路径。
        pt_file_path (str):  保存 PyTorch 模型的路径。
    """
    try:
        # 加载 ONNX 模型
        onnx_model = onnx.load(onnx_file_path)

        # 使用 onnx2pytorch 转换模型
        pytorch_model = ConvertModel(onnx_model)  # ConvertModel已经做了model.eval()

        # 保存 PyTorch 模型
        torch.save(pytorch_model, pt_file_path)

        print(f"成功: {onnx_file_path} 转换为 {pt_file_path}")

    except FileNotFoundError:
        print(f"错误: 文件 {onnx_file_path} 未找到。")
    except Exception as e:
        print(f"发生错误: {e}")

# 示例用法 (需要替换成你自己的文件路径)
if __name__ == '__main__':
    onnx_file = 'model.onnx'   # 替换成你的 ONNX 文件路径
    pt_file = 'model.pt'  # 替换成你想保存的 PyTorch 文件路径
    onnx_to_pt(onnx_file, pt_file)