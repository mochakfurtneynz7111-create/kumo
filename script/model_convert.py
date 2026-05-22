import os
import subprocess

def convert_model(input_model_path, output_pt_path, python_executable):
    """
    根据文件后缀名，调用相应的脚本进行模型转换。

    参数:
        input_model_path (str): 输入模型文件的路径。
        output_pt_path (str):  输出 PyTorch 模型的路径 (.pt 或 .pth)。
        python_executable (str): Python 解释器的路径 (例如: /usr/bin/python3, /path/to/your/venv/bin/python)。
    """
    _, file_extension = os.path.splitext(input_model_path)
    file_extension = file_extension.lower()
    script_path = os.path.join("convert")

    try:
        if file_extension == '.h5':
            script = os.path.join(script_path,"h5_to_pt.py")
            command = [python_executable, script, input_model_path, output_pt_path]
        elif file_extension == '.onnx':
            script = os.path.join(script_path,"onnx_to_pt.py")
            command = [python_executable, script, input_model_path, output_pt_path]
        elif file_extension == '.caffemodel':
            # 需要 prototxt 文件
             prototxt_path = input_model_path.replace(".caffemodel", ".prototxt") # Assumes same name
             if not os.path.exists(prototxt_path):
                 raise FileNotFoundError(f"找不到对应的 Prototxt 文件: {prototxt_path}")
             script = os.path.join(script_path,"caffe_to_pt.py")
             command = [python_executable, script, prototxt_path, input_model_path, output_pt_path]

        else:
            raise ValueError(f"不支持的文件格式: {file_extension}")

        # 调用子进程执行转换脚本
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            print(f"转换失败: {stderr.decode('utf-8')}")
            return None # 返回None 表示转换失败
        else:
            print(f"转换成功: {stdout.decode('utf-8')}")
            return output_pt_path # 返回输出文件路径

    except FileNotFoundError as e:
        print(f"文件未找到错误: {e}")
        return None
    except ValueError as e:
        print(f"值错误: {e}")
        return None
    except Exception as e:
        print(f"发生意外错误: {e}")
        return None

if __name__ == '__main__':
    # 示例用法 (替换为你自己的文件路径和 Python 解释器)
    input_model = 'test.onnx'  # 替换成你的输入模型文件
    output_model = 'test.pt'    # 替换成你想保存的 PyTorch 模型文件
    python_path = '/usr/bin/python3'  # 替换成你的 Python 解释器路径
    # python_path = 'D:\\anaconda3\\envs\\py38_torch110\\python.exe'

    result_path = convert_model(input_model, output_model, python_path)
    if result_path:
        print(f"转换后的模型保存在: {result_path}")
    else:
        print("模型转换失败。")