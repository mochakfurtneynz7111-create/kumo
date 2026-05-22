import os
import argparse
import subprocess
from pathlib import Path
import sys
import platform

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="递归编译指定目录下的所有.ui文件为PySide Python代码",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--ui-dir",
        type=str,
        default="ui",
        help=".ui文件所在的根目录路径"
    )
    parser.add_argument(
        "--gen-dir",
        type=str,
        default=None,
        help="生成的.py文件输出目录（默认与.ui文件同目录）"
    )
    parser.add_argument(
        "--pyside-version",
        type=str,
        choices=["pyside6", "pyside2"],
        default="pyside6",
        help="指定使用的PySide版本"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新生成所有文件（覆盖已存在的.py文件）"
    )
    # 添加新参数：uic工具绝对路径
    parser.add_argument(
        "--uic-path",
        type=str,
        default=None,
        help="手动指定uic工具的绝对路径（解决环境变量问题）"
    )
    return parser.parse_args()

def find_default_uic_path(pyside_version: str) -> str:
    """尝试在常见位置查找uic工具"""
    # Windows 默认路径
    if platform.system() == "Windows":
        appdata = os.getenv('APPDATA')
        if appdata:
            default_path = os.path.join(appdata, 'Python', 'Python38', 'Scripts', f'{pyside_version}-uic.exe')
            if os.path.exists(default_path):
                return default_path
    
    # Unix/Linux/Mac 默认路径
    python_bin = os.path.dirname(sys.executable)
    for path in [
        os.path.join(python_bin, f'{pyside_version}-uic'),
        os.path.join(python_bin, f'{pyside_version}-uic.exe')
    ]:
        if os.path.exists(path):
            return path
    
    return None

def compile_ui_files(ui_dir: str, gen_dir: str, pyside_version: str, force: bool, uic_path: str = None) -> None:
    """
    递归编译UI文件
    
    :param ui_dir: .ui文件所在目录
    :param gen_dir: 生成的.py文件输出目录
    :param pyside_version: PySide版本 ("pyside6" 或 "pyside2")
    :param force: 是否强制重新生成所有文件
    :param uic_path: uic工具的绝对路径
    """
    # 设置输出目录
    gen_dir = gen_dir if gen_dir is not None else ui_dir
    
    # 确定uic工具路径
    if uic_path:
        uic_tool = uic_path
    else:
        uic_tool = f"{pyside_version}-uic"
    
    # 检查工具是否可用
    try:
        subprocess.run([uic_tool, "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (FileNotFoundError, subprocess.CalledProcessError):
        # 尝试自动查找默认路径
        found_path = find_default_uic_path(pyside_version)
        if found_path:
            print(f"⚠️  使用自动发现的uic路径: {found_path}")
            uic_tool = found_path
        else:
            raise ImportError(
                f"未找到{uic_tool}，请先安装: pip install {pyside_version}\n"
                "或使用 --uic-path 参数指定绝对路径"
            )

    # 递归处理.ui文件
    for root, _, files in os.walk(ui_dir):
        for file in files:
            if file.endswith(".ui"):
                ui_path = Path(root) / file
                relative_dir = os.path.relpath(root, ui_dir)
                output_dir = Path(gen_dir) / relative_dir
                output_dir.mkdir(parents=True, exist_ok=True)
                
                output_file = output_dir / f"{ui_path.stem}_ui.py"
                
                if not force and output_file.exists():
                    print(f"⏩ 已存在: {output_file}（使用--force覆盖）")
                    continue
                
                try:
                    subprocess.run(
                        [uic_tool, str(ui_path), "-o", str(output_file)],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    print(f"✅ 成功编译: {ui_path} -> {output_file}")
                except subprocess.CalledProcessError as e:
                    error_msg = e.stderr.decode().strip() if e.stderr else str(e)
                    print(f"❌ 编译失败: {ui_path}\n错误信息: {error_msg}")
                    sys.exit(1)

if __name__ == "__main__":
    args = parse_args()
    try:
        compile_ui_files(
            ui_dir=args.ui_dir,
            gen_dir=args.gen_dir,
            pyside_version=args.pyside_version,
            force=args.force,
            uic_path=args.uic_path  # 传递新的uic-path参数
        )
    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)