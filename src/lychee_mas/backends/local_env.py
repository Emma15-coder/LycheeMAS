"""本地.env文件权限校验；语法解析和环境加载交给python-dotenv。"""
from __future__ import annotations

import os
import stat
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Iterator


@contextmanager
def checked_env_stream(path: Path, *, optional: bool = False) -> Iterator[StringIO | None]:
    """安全打开已指定文件并检查600权限；不查找父目录.env，不打印内容。"""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        if optional and not path.is_symlink():
            yield None
            return
        raise
    with os.fdopen(fd, "r", encoding="utf-8") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise ValueError("本地配置必须为当前用户拥有的600普通文件。")
        text = stream.read(16385)
    if len(text) > 16384:
        raise ValueError("本地配置超过长度限制。")
    with StringIO(text) as content:
        yield content
