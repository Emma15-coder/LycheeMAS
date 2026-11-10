"""可选 Searcher 诊断日志；默认无输出，不保存文件，不回显请求头/异常正文。"""
from __future__ import annotations

import json
import logging
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any
from urllib.parse import quote

LOGGER = logging.getLogger("lychee_mas.mindsearch.trace")
RUN_ID: ContextVar[str] = ContextVar("searcher_run_id", default="")
STAGE: ContextVar[dict] = ContextVar("searcher_stage")


@contextmanager
def trace_file(path: Path, *, enabled: bool = False):
    """显式启用的脱敏诊断文件；关闭后恢复logger，避免重复运行累积handler。"""
    if not enabled:
        yield
        return
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
    os.fchmod(fd, 0o600)
    previous = LOGGER.level
    with os.fdopen(fd, "a", encoding="utf-8") as stream:
        handler = logging.StreamHandler(stream)
        LOGGER.addHandler(handler)
        LOGGER.setLevel(logging.INFO)
        try:
            yield
        finally:
            LOGGER.removeHandler(handler)
            handler.close()
            LOGGER.setLevel(previous)


def redact(value: str) -> str:
    """隐藏环境中的凭据、常见Key格式、Bearer及URL中的敏感查询参数。"""
    secrets = [v for k, v in os.environ.items() if v and re.search(
        r"(?:API_KEY|TOKEN|SECRET|PASSWORD)", k, re.IGNORECASE)]
    for secret in sorted(secrets, key=len, reverse=True):
        for variant in (secret, json.dumps(secret, ensure_ascii=False)[1:-1],
                        json.dumps(secret, ensure_ascii=True)[1:-1], quote(secret, safe="")):
            if variant:
                value = value.replace(variant, "[REDACTED]")
    value = re.sub(r"(?i)Bearer\s+[^\s\"'<>\\]+", "Bearer [REDACTED]", value)
    value = re.sub(r"\b(?:sk-|tvly-)[A-Za-z0-9_-]+", "[REDACTED]", value)
    value = re.sub(r"(?i)([?&](?:api[_-]?key|token|access_token|secret|password)=)[^&#\s\"'<>\\]*",
                   r"\1[REDACTED]", value)
    return value


def redact_fields(value: Any) -> Any:
    """逐字段脱敏后再序列化，避免正则吞掉JSON引号或转义符。"""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: redact_fields(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_fields(v) for v in value]
    return value


def trace(event: str, **fields: Any) -> None:
    """换行/终端控制符保持JSON转义；调用方限制字段大小。"""
    if LOGGER.isEnabledFor(logging.INFO):
        data = {"event": event, "run_id": RUN_ID.get(), **fields}
        LOGGER.info("%s", json.dumps(redact_fields(data), ensure_ascii=False))
