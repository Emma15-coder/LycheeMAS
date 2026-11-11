"""让离线测试可按实际命令行入口的方式导入scripts中的同目录辅助模块。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
