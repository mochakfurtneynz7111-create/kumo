import subprocess
import sys
from pathlib import Path
import os
import shutil
import importlib.util
from typing import List, Union, Optional

ui_dir = "ui"  # .ui文件所在目录
executable_dir = "build"
# 需要复制的文件/文件夹列表（相对路径）
FILES_TO_COPY = [
    "style/style.qss",
    "src/icon/",
    "setting.json",
    "script/",
    "netron/",
    "ui_params/",
    "result/infer.json"
]

# 检测PySide版本
def detect_pyside_version():
    """通过实际导入验证版本"""
    try:
        import PySide6
        return "pyside6"
    except ImportError:
        try:
            import PySide2
            return "pyside2"
        except ImportError:
            raise ImportError("Neither PySide6 nor PySide2 is installed")


def run_command(command: Union[str, List[str]], cwd: str = None) -> bool:
    """执行命令并实时输出日志（解决编码问题）
    
    Args:
        command: 命令字符串或参数列表
        cwd: 工作目录路径
    Returns:
        bool: 是否执行成功
    """
    try:
        # 统一处理命令格式
        if isinstance(command, str) and sys.platform != "win32":
            command = ["sh", "-c", command]
        elif isinstance(command, str):
            command = ["cmd", "/c", command]

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            shell=False,  # 显式禁用shell模式更安全[7,8](@ref)
            universal_newlines=False  # 禁用自动文本解码[1](@ref)
        )

        # 实时处理输出流
        while True:
            raw_output = process.stdout.readline()
            if not raw_output and process.poll() is not None:
                break
            
            try:
                # 尝试UTF-8解码（覆盖大多数现代程序）
                output = raw_output.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    # 回退到系统默认编码（Windows为GBK）
                    output = raw_output.decode(sys.getdefaultencoding(), errors='replace')
                except:
                    # 终极回退方案：替换所有非法字符
                    output = raw_output.decode('utf-8', errors='replace')
                    
            if output:
                print(output.strip())

        return process.returncode == 0

    except Exception as e:
        print(f"❌ 命令执行失败: {e}")
        return False

def copy_with_structure(src_root: Path, dst_root: Path, items: List[str]) -> None:
    """
    保留目录结构复制文件/文件夹
    :param src_root: 项目根目录
    :param dst_root: 目标根目录（如out/main.dist）
    :param items: 需要复制的文件/文件夹列表
    """
    for item in items:
        src_path = src_root / item
        dst_path = dst_root / item

        try:
            if src_path.is_file():
                # 创建目标目录（如果不存在）
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                # 复制文件（保留元数据）
                shutil.copy2(src_path, dst_path)
                print(f"📄 复制文件: {src_path} -> {dst_path}")
            elif src_path.is_dir():
                # 复制整个目录树
                shutil.copytree(
                    src_path, 
                    dst_path,
                    dirs_exist_ok=True,  # 允许目标目录存在
                    copy_function=shutil.copy2  # 保留元数据
                )
                print(f"📂 复制目录: {src_path} -> {dst_path}")
            else:
                print(f"⚠️ 路径不存在: {src_path}")
        except Exception as e:
            print(f"❌ 复制失败 [{src_path}]: {e}")

def main():
    # 1. 检测PySide版本
    print("="*50)
    pyside_version = detect_pyside_version()
    print(f"☑️ PySide版本: {pyside_version}")
    # 2. 执行UI编译脚本
    print("="*50)
    print("🛠️ 开始编译UI文件...")
    ui_script = Path("build_tools") / "build_ui.py"
    ui_cmd = [
        "python",
        str(ui_script),
        "--ui-dir",
        ui_dir,
        "--pyside-version",
        pyside_version,
        "--force"
    ]
    if not run_command(" ".join(ui_cmd)):
        # print(f'命令：{" ".join(ui_cmd)}')
        print("❌ UI编译失败，终止流程")
        sys.exit(1)

    # 3. 执行Nuitka编译
    print("\n" + "="*50)
    print("🔧 开始Nuitka编译...")
    nuitka_cmd = [
        "nuitka",
        "--standalone",
        "--show-memory",
        "--show-progress",
        # "--follow-imports",
        "--nofollow-imports",
        "--follow-import-to=netron",
        "--follow-import-to=src.QTCompat",
        # "--include-package=QTCompat",
        f"--output-dir={executable_dir}",
        # "--windows-disable-console",
        f"--enable-plugin={pyside_version}",
        "main.py"
    ]
    if not run_command(" ".join(nuitka_cmd)):
        print("❌ Nuitka编译失败")
        sys.exit(1)

	# 4. 复制资源文件
    print("\n" + "="*50)
    print("📦 复制资源文件...")
    dist_dir = Path(executable_dir) / "main.dist"
    copy_with_structure(Path("."), dist_dir, FILES_TO_COPY)
    print(f"\n✅ 所有步骤完成！输出文件位于: {executable_dir}/main.dist/ 目录")

if __name__ == "__main__":
    main()