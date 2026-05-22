import os
import subprocess
# from src.QTCompat import QProcess
import sys
import glob
import logging

logger = logging.getLogger(__name__)
# import time

def findPythonInterpreters(search_paths=None):
    """查找系统中所有Python解释器的位置"""
    python_paths = []

    if sys.platform == 'win32':
        # Windows: 检查常见路径和where命令
        common_paths = [
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python*\python.exe"),
            os.path.expandvars(r"%PROGRAMFILES%\Python\Python*\python.exe"),
            os.path.expandvars(r"%PROGRAMFILES(x86)%\Python\Python*\python.exe"),
        ]
        
        # 使用where命令查找python.exe
        try:
            result = subprocess.run(
                ['where', 'python.exe'], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                shell=True
            )
            python_paths.extend(result.stdout.splitlines())
        except FileNotFoundError:
            pass

    else:
        # Linux/Unix: 检查常见路径和which命令
        common_paths = [
            '/usr/bin/python*',
            '/usr/local/bin/python*',
            '/opt/python*/bin/python*',
            os.path.expanduser('~/.local/bin/python*'),
        ]

        # 使用which命令查找所有python和python3
        for cmd in ['python', 'python3']:
            try:
                result = subprocess.run(
                    ['which', '-a', cmd], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True
                )
                python_paths.extend(result.stdout.splitlines())
            except FileNotFoundError:
                pass

    # 检查所有常见路径
    for pattern in common_paths:
        for path in glob.glob(pattern):
            if os.path.isfile(path) and isPythonInterpreter(path):
                python_paths.append(path)
                
    # 搜索用户传入路径
    if search_paths:
        for path in search_paths:
            print(f"Searching in {path}")
            for root, dirs, files in os.walk(path):
                for file in files:
                    if isPythonInterpreter(file):
                        python_paths.append(os.path.join(root, file))

    # 去重并返回
    return sorted(set(os.path.realpath(p) for p in python_paths if p))

def isPythonInterpreter(filename):
    """判断一个文件名是否是Python解释器"""
    name = filename.lower()
    if sys.platform == 'win32':
        return name == 'python.exe' or name.startswith('python') and name.endswith('.exe')
    else:
        return name == 'python' or name == 'python2' or name == 'python3' or name.startswith('python')

def getPythonVersion(path):
    try:
        python_version = subprocess.run([path, '--version'],
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True, shell=True).stdout.strip()
    except Exception as e:
        logger.error(f"Error getting version for {path}: {e}")
        return ""
    
    return python_version

# def getPythonVersion(path):
#     process = QProcess()
#     process.start(path, ['--version'])
#     process.waitForFinished()  # 可选：同步等待完成
#     output = process.readAllStandardOutput().data().decode().strip()
#     process.close()  # 显式释放资源
#     return output

def getPythonVersionList(search_paths=None, extra_paths=None):
    python_paths = findPythonInterpreters(search_paths)
    if extra_paths:
        python_paths.extend(extra_paths)
    unique_lst = []
    for x in python_paths:
        if x not in unique_lst:
            unique_lst.append(x)
    
    python_paths = unique_lst
    python_version_list = []
    for path in python_paths:
        try:
            version = getPythonVersion(path)
            if version.startswith('Python'):
                python_version_list.append((version, path))
        except Exception as e:
            logger.error(f"Error getting version for {path}: {e}")
    # time.sleep(2)
    return python_version_list

if __name__ == "__main__":
    interpreters = getPythonVersionList()
    print("\nFound Python interpreters:")
    for i, path in enumerate(interpreters, 1):
        print(f"{i}. {path}")
    
    if not interpreters:
        print("No Python interpreters found.")