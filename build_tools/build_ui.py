import os
import argparse
import subprocess
from pathlib import Path
import sys

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="递归编译指定目录下的所有.ui文件为PySide Python代码",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # 显示默认值
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
    return parser.parse_args()

def compile_ui_files(ui_dir: str, gen_dir: str, pyside_version: str, force: bool) -> None:
    """
    递归编译UI文件
    
    :param ui_dir: .ui文件所在目录
    :param gen_dir: 生成的.py文件输出目录
    :param pyside_version: PySide版本 ("pyside6" 或 "pyside2")
    :param force: 是否强制重新生成所有文件
    """
    # 设置输出目录（若未指定则与.ui目录相同）
    gen_dir = gen_dir if gen_dir is not None else ui_dir
    
    # 获取uic工具路径
    uic_tool = f"{pyside_version}-uic"
    try:
        subprocess.run([uic_tool, "--version"], check=True, capture_output=True)
    except FileNotFoundError:
        raise ImportError(f"未找到{uic_tool}，请先安装: pip install {pyside_version}")

    # 递归处理.ui文件
    for root, _, files in os.walk(ui_dir):
        for file in files:
            if file.endswith(".ui"):
                ui_path = Path(root) / file
                relative_dir = os.path.relpath(root, ui_dir)
                output_dir = Path(gen_dir) / relative_dir
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # 生成.py文件名（保留原文件名+_ui后缀）
                output_file = output_dir / f"{ui_path.stem}_ui.py"
                
                # 跳过已存在且不需要强制重写的文件
                if not force and output_file.exists():
                    print(f"已存在: {output_file}（使用--force覆盖）")
                    continue
                
                # 调用uic工具
                try:
                    subprocess.run(
                        [uic_tool, str(ui_path), "-o", str(output_file)],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    print(f"成功编译: {ui_path} -> {output_file}")
                except subprocess.CalledProcessError as e:
                    print(f"编译失败: {ui_path}\n错误信息: {e.stderr.decode().strip()}")
                    sys.exit(1)

if __name__ == "__main__":
    args = parse_args()
    try:
        compile_ui_files(
            ui_dir=args.ui_dir,
            gen_dir=args.gen_dir,
            pyside_version=args.pyside_version,
            force=True
        )
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)