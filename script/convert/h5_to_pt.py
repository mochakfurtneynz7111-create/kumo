import h5py
import torch
import numpy as np  # 确保安装 numpy

def h5_to_pt(h5_file_path, pt_file_path):
    """
    将 HDF5 文件转换为 PyTorch 张量 (.pt 文件).

    参数:
        h5_file_path (str):  HDF5 文件的路径。
        pt_file_path (str):  保存 PyTorch 张量的路径。
    """
    try:
        with h5py.File(h5_file_path, 'r') as hf:
            # 假设 HDF5 文件中只有一个数据集，名为 'data'
            # 如果你的 HDF5 文件结构不同，你需要修改这段代码
            if 'data' in hf:
                data = hf['data'][:]  # 读取数据集到 NumPy 数组
            else:
                print(f"错误: HDF5 文件 {h5_file_path} 中找不到名为 'data' 的数据集。")
                return

            # 将 NumPy 数组转换为 PyTorch 张量
            tensor = torch.from_numpy(data)

            # 保存 PyTorch 张量到文件
            torch.save(tensor, pt_file_path)

            print(f"成功:  {h5_file_path} 转换为 {pt_file_path}")

    except FileNotFoundError:
        print(f"错误:  文件 {h5_file_path} 未找到。")
    except Exception as e:
        print(f"发生错误: {e}")

# 示例用法 (需要替换成你自己的文件路径)
if __name__ == '__main__':
    h5_file = 'input.h5'   # 替换成你的 HDF5 文件路径
    pt_file = 'output.pt'  # 替换成你想保存的 PyTorch 文件路径
    h5_to_pt(h5_file, pt_file)