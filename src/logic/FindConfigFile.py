import os
from pathlib import Path
import time

def getConfigFileList(directory, num=5):
    """
    获取目录下最新的N个文件
    :param directory: 目标路径
    :param num: 要获取的文件数量（默认5）
    :return: 按修改时间降序排列的文件路径列表
    """
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        return []
    files = [f for f in dir_path.iterdir() if f.is_file()]
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    # time.sleep(2)
    if num  < 0:
        num = len(files)
    return [str(f) for f in files[:num]]


if __name__ == "__main__":
	# 示例：获取/home/docs下最新的5个文件
	latest_files = getConfigFileList("/home/docs")
	for file in latest_files:
		print(file)