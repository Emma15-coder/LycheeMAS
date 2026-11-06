"""Tavily Search/Extract 薄适配；标准库 HTTP，无 SDK 重试、无 Research/答案生成。

API: https://docs.tavily.com/documentation/api-reference/endpoint/search
     https://docs.tavily.com/documentation/api-reference/endpoint/extract
"""
from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import Any

from .searcher_trace import trace


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> None:
        return None


class TavilyBackend:
    """search 每条 query 一次 POST；fetch 每条 URL 一次 POST。失败不重试。

    预算跨每次 search 调用：最多3个查询，每条最多10条结果；
    提取正文最多300000字符，模型输入预算由Searcher单独限制。
    网络错误只暴露 HTTP 状态/异常类别，避免回显 Key 或响应正文。
    """

    def __init__(self, *, api_key_env: str = "TAVILY_API_KEY", timeout: float = 20.0,
                 max_page_chars: int = 300000) -> None:
        self._key = os.environ.get(api_key_env, "").strip()
        if not self._key:
            raise ValueError("Tavily API key environment variable is missing")
        if not 0 < timeout <= 30 or not 1 <= max_page_chars <= 300000:
            raise ValueError("invalid Tavily budget")
        self.timeout = timeout
        self.max_page_chars = max_page_chars
        self._opener = urllib.request.build_opener(_NoRedirect())

    def _post(self, endpoint: str, body: dict) -> dict:
        request = urllib.request.Request(
            f"https://api.tavily.com/{endpoint}",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self._key}",
                     "Content-Type": "application/json"}, method="POST")
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                if response.status != 200:
                    raise ValueError("non-200 response")
                raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ValueError("response exceeds byte budget")
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict) or "error" in data or "detail" in data:
                raise ValueError("invalid response")
            return data
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Tavily HTTP {exc.code}; details redacted") from None
        except Exception as exc:
            raise RuntimeError(f"Tavily {type(exc).__name__}; details redacted") from None

    async def search(self, queries: list[str], topk: int) -> list[dict]:
        if not 1 <= len(queries) <= 3 or type(topk) is not int or not 1 <= topk <= 10:
            raise ValueError("invalid Tavily search budget")
        if any(not isinstance(q, str) or not q.strip() or len(q) > 256 for q in queries):
            raise ValueError("invalid Tavily query")
        rows = []
        # 逐条请求，失败立即停止；线程避免阻塞其它 Searcher/Planner 任务。
        for query in dict.fromkeys(queries):
            data = await asyncio.to_thread(self._post, "search", {
                "query": query, "search_depth": "basic", "max_results": topk,
                "include_answer": False, "include_raw_content": False,
                "auto_parameters": False,
            })
            results = data.get("results")
            if not isinstance(results, list):
                raise RuntimeError("Tavily search response missing results")
            for item in results[:topk]:
                if not isinstance(item, dict) or any(
                    not isinstance(item.get(k), str) for k in ("url", "title", "content")
                ):
                    raise RuntimeError("Tavily search item has invalid fields")
                rows.append({"url": item["url"], "title": item["title"][:500],
                             "snippet": item["content"][:1500]})
        return rows

    async def fetch(self, url: str) -> str:
        # URL 验证在 Searcher 合并层和本适配器边界各执行一次。
        from .web_utils import canonical_url

        if not canonical_url(url):
            raise ValueError("invalid fetch URL")
        data = await asyncio.to_thread(self._post, "extract", {
            "urls": [url], "extract_depth": "basic", "format": "markdown",
        })
        if data.get("failed_results"):
            raise RuntimeError("Tavily extract failed; details redacted")
        rows = data.get("results")
        if not isinstance(rows, list):
            raise RuntimeError("Tavily extract response missing results")
        if not rows:
            return ""
        if (len(rows) != 1 or not isinstance(rows[0], dict)
                or canonical_url(rows[0].get("url")) != canonical_url(url)
                or not isinstance(rows[0].get("raw_content"), str)):
            raise RuntimeError("Tavily extract response mismatch")
        content = rows[0]["raw_content"]
        trace("tavily.extract", url=url, raw_chars=len(content),
              returned_chars=min(len(content), self.max_page_chars),
              truncated=len(content) > self.max_page_chars)
        return content[:self.max_page_chars]
