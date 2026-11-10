"""单子问题 Searcher 冒烟；无 Planner、无文件落盘，详见 docs/mindsearch-searcher.md。"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict

from lychee_mas.backends.async_chat import AsyncChat
from lychee_mas.backends.openai_api_backend import OpenAICompatibleBackend
from lychee_mas.backends.searcher_trace import LOGGER, redact_fields
from lychee_mas.backends.tavily_backend import TavilyBackend
from lychee_mas.methods.build.mindsearch.searcher import WebSearcher


def make_searcher() -> WebSearcher:
    """复用现有后端；显式端点、Key环境变量，关闭 SDK 重试与重定向。"""
    tavily = TavilyBackend()
    backend = OpenAICompatibleBackend(
        model=os.environ.get("LYCHEE_LLM_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash"),
        base_url=os.environ.get("LYCHEE_LLM_BASE_URL", "https://api.siliconflow.cn/v1"),
        api_key_env="SILICONFLOW_API_KEY", timeout=30.0, temperature=0.0,
    )
    import httpx  # openai 的传递依赖；不在注册链导入

    original = backend.client
    backend.client = original.with_options(
        max_retries=0, http_client=httpx.Client(timeout=30.0, follow_redirects=False))
    original.close()
    return WebSearcher(AsyncChat(backend, max_new_tokens=512), tavily.search, tavily.fetch,
                       topk=3, max_queries=1, max_read=1)


async def run(question: str) -> int:
    start = time.perf_counter()
    try:
        searcher = make_searcher()
        try:
            result = await searcher.run(question)
        finally:
            searcher.llm.backend.client.close()
    except Exception as exc:
        print(f"Setup failed ({type(exc).__name__}); details redacted")
        return 1
    # SDK错误绝不输出原文；额外防御服务端在内容中反射凭据。
    output = json.dumps(redact_fields(asdict(result)), ensure_ascii=False, indent=2)
    print(output)
    print(f"elapsed_s={time.perf_counter() - start:.3f}")
    return 0 if result.is_ok() else 1


def main() -> None:
    # 只打开经过脱敏的专用日志；不暴露SDK/HTTP请求调试输出。
    logging.disable(logging.NOTSET)
    os.environ.pop("OPENAI_LOG", None)
    for name in ("openai", "httpx", "httpcore"):
        sdk_logger = logging.getLogger(name)
        sdk_logger.handlers.clear()
        sdk_logger.addHandler(logging.NullHandler())
        sdk_logger.propagate = False
        sdk_logger.disabled = True
    LOGGER.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default="居里夫人出生在哪个城市，该城市现在属于哪个国家？")
    args = parser.parse_args()
    print("调用上限：模型3次（每次<=512输出tokens），Tavily Search 1次，Extract 1次；无重试。")
    raise SystemExit(asyncio.run(run(args.question)))


if __name__ == "__main__":
    main()
