"""将同步生成后端适配为异步 prompt 回调，保留文本/token/延迟返回对象。"""
from __future__ import annotations

import asyncio
from typing import Any


class AsyncChat:
    """生成在工作线程执行，不阻塞事件循环；后端必须自行设置传输超时/重试。

    wait_for 的取消不会终止已经运行的线程，真实 HTTP 仍由后端传输超时收尾。
    不自动重试；不捕获外部 CancelledError。
    """

    def __init__(self, backend: Any, *, max_new_tokens: int = 512,
                 timeout: float = 35.0) -> None:
        if max_new_tokens < 1 or timeout <= 0:
            raise ValueError("invalid async chat budget")
        self.backend = backend
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout

    async def __call__(self, prompt: str) -> Any:
        return await asyncio.wait_for(asyncio.to_thread(
            self.backend.generate_chat, [{"role": "user", "content": prompt}],
            max_new_tokens=self.max_new_tokens), timeout=self.timeout)
