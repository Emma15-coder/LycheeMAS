"""本地密钥安装/检查与已授权单题冒烟；不接受任意命令或批量任务。"""
from __future__ import annotations

import argparse
import getpass
import os
import stat
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEY_NAMES = ("SILICONFLOW_API_KEY", "TAVILY_API_KEY")
QUESTION = "居里夫人出生在哪个城市，该城市现在属于哪个国家？"


class _SecretConfigError(ValueError):
    """只携带本脚本定义的固定诊断，不携带文件内容。"""


def _location() -> Path:
    return Path.home() / ".config" / "lychee-mas" / "secrets.env"


def _check_directory(directory: Path) -> None:
    info = directory.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise _SecretConfigError("密钥目录必须由当前用户拥有、非符号链接且权限为700。")
    if directory.resolve().is_relative_to(ROOT):
        raise _SecretConfigError("密钥目录必须位于项目外。")


def _validate_value(value: str) -> None:
    if not value or len(value) > 4096 or any(c.isspace() or not c.isprintable() for c in value):
        raise _SecretConfigError("Key必须非空，且不能包含空白或控制字符。")


def _load(path: Path) -> dict[str, str]:
    _check_directory(path.parent)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "r", encoding="utf-8") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise _SecretConfigError("密钥文件必须由当前用户拥有、为普通文件且权限为600。")
        text = stream.read(16385)
    if len(text) > 16384:
        raise _SecretConfigError("密钥文件超过长度限制。")
    keys = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        if not separator or name not in KEY_NAMES or name in keys:
            raise _SecretConfigError("密钥文件格式无效：仅允许两项指定Key，不允许重复。")
        _validate_value(value)
        keys[name] = value
    if set(keys) != set(KEY_NAMES):
        raise _SecretConfigError("密钥文件缺少指定Key。")
    return keys


def _setup(path: Path) -> None:
    if path.parent.resolve().is_relative_to(ROOT):
        raise _SecretConfigError("密钥目录必须位于项目外。")
    # 无可用TTY时禁止getpass退化成明文输入。
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        keys = {name: getpass.getpass(f"{name}（隐藏输入）: ") for name in KEY_NAMES}
    for value in keys.values():
        _validate_value(value)
    path.parent.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.mkdir(mode=0o700, exist_ok=True)
    info = path.parent.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise _SecretConfigError("密钥目录所有者或类型无效。")
    path.parent.chmod(0o700)
    _check_directory(path.parent)
    if path.is_symlink():
        raise _SecretConfigError("拒绝覆盖符号链接。")
    fd, temporary = tempfile.mkstemp(prefix=".secrets-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write("".join(f"{name}={keys[name]}\n" for name in KEY_NAMES))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print("密钥已保存至 ~/.config/lychee-mas/secrets.env；目录700，文件600。")


def _smoke(keys: dict[str, str]) -> None:
    python = ROOT / ".venv" / "bin" / "python"
    if not python.is_file():
        raise _SecretConfigError("项目.venv/bin/python不存在；未发送请求。")
    env = dict(os.environ)
    env.update(keys)
    env.update({
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "LYCHEE_LLM_BASE_URL": "https://api.siliconflow.cn/v1",
        "LYCHEE_LLM_MODEL": "deepseek-ai/DeepSeek-V4.1-Flash",
    })
    env.pop("OPENAI_LOG", None)
    # 密钥不进入命令行参数；仅在目标进程环境中传递。
    os.execve(python, [str(python), str(ROOT / "scripts" / "smoke_mindsearch_searcher.py"),
                      "--question", QUESTION], env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("setup", "check", "smoke"), default="smoke")
    args = parser.parse_args()
    try:
        path = _location()
        if args.mode == "setup":
            _setup(path)
        else:
            keys = _load(path)
            if args.mode == "check":
                print("密钥文件格式、所有权与权限检查通过；未发送请求。")
            else:
                _smoke(keys)
    except _SecretConfigError as exc:
        # ValueError只来自本文件中的固定诊断，不包含密钥内容。
        print(str(exc), file=sys.stderr)
        return 1
    except (OSError, UnicodeError, getpass.GetPassWarning, EOFError, ValueError):
        print("本地密钥读取/保存失败；请检查文件、权限及终端，未显示敏感详情。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
