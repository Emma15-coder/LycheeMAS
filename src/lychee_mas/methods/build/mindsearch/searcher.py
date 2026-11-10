"""WebSearcher —— 分层检索流水（B 负责实现，签名与语义在此冻结）。

四步：① 多 query 改写 ② 按 URL 合并搜索结果 ③ 选页读正文 ④ 总结回填。

三条冻结约定（改动需 A/B 双方同意并单独提交 refactor）：
1. 失败不抛异常：网络/抓取失败返回 status="error" + error 文本，run() 内部最多重试 1 次；
   由 planner 决定是重规划还是 mark_failed——本层不假装成功，也不抛穿整个 run。
2. 空结果：检索到了但没有可用内容 -> status="empty"、answer=""。
3. context 语义：固定为「父节点 answer + 分隔线 + 根问题」，不含兄弟节点与全部历史。
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ....backends.search_errors import SearchBackendError
from ....backends.searcher_trace import RUN_ID, STAGE, trace
from ....backends.web_utils import canonical_url
from .page_content import resolve_evidence_quote, select_page_content
from .searcher_prompts import REWRITE_PROMPT, SELECT_PROMPT, SUMMARIZE_PROMPT

LLM = Callable[[str], Awaitable[Any]]
# (queries, topk) -> [{url,title,snippet}]
SearchFn = Callable[[list[str], int], Awaitable[list[dict]]]
FetchFn = Callable[[str], Awaitable[str]]                      # (url) -> 正文纯文本


@dataclass(frozen=True)
class SourceRef:
    """检索来源。sources 里只允许这三种键（to_dict 即落盘格式）。"""

    url: str
    title: str = ""
    snippet: str = ""

    def to_dict(self) -> dict:
        return {"url": self.url, "title": self.title, "snippet": self.snippet}


@dataclass
class SearcherResult:
    sub_question: str
    status: str = "ok"                     # ok | empty | error
    answer: str = ""
    sources: list[SourceRef] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    pages_read: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""                        # status=="error" 时必填

    def is_ok(self) -> bool:
        return self.status == "ok"


class WebSearcher:
    """分层检索器。依赖全部注入，不绑定任何搜索 SDK。"""

    def __init__(self, llm: LLM, search: SearchFn, fetch: FetchFn, *,
                 topk: int = 3, max_queries: int = 3, max_read: int = 3) -> None:
        self.llm, self.search, self.fetch = llm, search, fetch
        self.topk, self.max_queries, self.max_read = topk, max_queries, max_read

    # 固定安全预算，避免修改冻结构造签名。自定义服务需在回调中设置更短传输超时。
    REQUEST_TIMEOUT_S = 35.0
    MODEL_TIMEOUT_S = 35.0
    RUN_TIMEOUT_S = 180.0
    MAX_PAGE_CHARS = 12000
    MAX_FETCH_CHARS = 300000
    MAX_INPUT_CHARS = 8000
    MAX_QUERY_CHARS = 256
    MAX_LLM_CHARS = 32000

    async def run(self, sub_question: str, context: str = "") -> SearcherResult:
        """四阶段检索；无内部重试（0 <= 契约上限1），状态/统计按每次调用隔离。

        context 原样解释为父节点答案 + 分隔线 + 根问题，不扩充其它历史。
        网络/模型/格式错误返回 error；无网页、正文或支持证据返回 empty。
        外部取消仍向上传播。pages_read 只计成功取得非空正文的页面。
        """
        result = SearcherResult(sub_question=sub_question)
        token = RUN_ID.set(uuid.uuid4().hex[:12])
        stage_token = STAGE.set({"name": "validate"})
        started = time.perf_counter()
        trace("searcher.start", sub_question=sub_question)
        try:
            if not isinstance(sub_question, str) or not sub_question.strip():
                raise ValueError("invalid sub-question")
            if not isinstance(context, str):
                raise ValueError("invalid context")
            if max(len(sub_question), len(context)) > self.MAX_INPUT_CHARS:
                raise ValueError("input budget exceeded")
            if self.max_queries > 3:
                raise ValueError("query budget exceeds adapter limit")
            if any(type(x) is not int or x < 1 or x > 10
                   for x in (self.topk, self.max_queries, self.max_read)):
                raise ValueError("invalid search budget")
            await asyncio.wait_for(self._run(result, context), timeout=self.RUN_TIMEOUT_S)
        except Exception as exc:
            # 不回显回调异常（SDK/HTTP 异常可能含认证信息或请求正文）。
            result.status = "error"
            result.answer = ""
            result.sources = []
            result.error = f"searcher failed ({type(exc).__name__}); details redacted"
            # 只公开本模块已知的静态错误码，绝不回显外部回调的异常正文。
            known = {"invalid source IDs", "invalid summary", "answer without sources",
                     "answer citations mismatch", "invalid evidence", "unsupported evidence",
                     "missing evidence", "incomplete model response", "invalid model output",
                     "expected JSON object", "invalid queries", "invalid page content"}
            reason = ("invalid_json" if isinstance(exc, json.JSONDecodeError) else
                      "timeout" if isinstance(exc, TimeoutError) else
                      str(exc).replace(" ", "_") if type(exc) is ValueError and str(exc) in known
                      else "external_or_validation_error")
            result.error += f"; stage={STAGE.get()['name']}; reason={reason}"
            if isinstance(exc, SearchBackendError):
                reason = exc.reason
                result.error = (f"searcher failed ({type(exc).__name__}); details redacted; "
                                f"stage={STAGE.get()['name']}; reason={reason}")
                if exc.http_status is not None:
                    result.error += f"; http_status={exc.http_status}"
            trace("searcher.error", error_type=type(exc).__name__,
                  stage=STAGE.get()["name"], reason=reason,
                  http_status=exc.http_status if isinstance(exc, SearchBackendError) else None)
        finally:
            trace("searcher.end", status=result.status, pages_read=result.pages_read,
                  prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens,
                  elapsed_s=round(time.perf_counter() - started, 3))
            RUN_ID.reset(token)
            STAGE.reset(stage_token)
        return result

    async def _call(self, fn: Callable, *args: Any, timeout: float | None = None) -> Any:
        return await asyncio.wait_for(fn(*args), timeout=(
            self.REQUEST_TIMEOUT_S if timeout is None else timeout))

    async def _json(self, prompt: str, result: SearcherResult, stage: str) -> dict:
        STAGE.get()["name"] = stage
        started = time.perf_counter()
        trace("model.start", stage=stage, prompt_chars=len(prompt), prompt=prompt)
        response = await self._call(self.llm, prompt, timeout=self.MODEL_TIMEOUT_S)
        # 支持既有 APIGenResult/GenResult，及离线裸字符串回调。
        result.prompt_tokens += int(getattr(response, "n_prompt_pos", 0))
        result.completion_tokens += int(getattr(response, "n_gen_tokens", 0))
        text = getattr(response, "text", response)
        finish_reason = getattr(response, "finish_reason", None)
        trace("model.response", stage=stage,
              text=text[:self.MAX_LLM_CHARS] if isinstance(text, str) else None,
              finish_reason=finish_reason,
              prompt_tokens=int(getattr(response, "n_prompt_pos", 0)),
              completion_tokens=int(getattr(response, "n_gen_tokens", 0)),
              elapsed_s=round(time.perf_counter() - started, 3))
        if finish_reason in ("length", "content_filter"):
            trace("model.incomplete", stage=stage, finish_reason=finish_reason)
            raise ValueError("incomplete model response")
        if not isinstance(text, str) or len(text) > self.MAX_LLM_CHARS:
            raise ValueError("invalid model output")
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("expected JSON object")
        return data

    @staticmethod
    def _empty(result: SearcherResult, reason: str) -> None:
        result.status = "empty"
        trace("searcher.empty", reason=reason, pages_read=result.pages_read)

    @staticmethod
    def _ids(data: dict, allowed: dict) -> list[str]:
        ids = data.get("source_ids")
        if not isinstance(ids, list) or any(not isinstance(i, str) or i not in allowed
                                            for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("invalid source IDs")
        return ids

    async def _run(self, result: SearcherResult, context: str) -> None:
        def dump(data: Any) -> str:
            return json.dumps(data, ensure_ascii=False)

        base = {"sub_question": result.sub_question, "context": context}
        data = await self._json(REWRITE_PROMPT.format(
            limit=self.max_queries, data=dump(base)), result, "rewrite")
        queries = data.get("queries")
        if not isinstance(queries, list) or not queries or any(
            not isinstance(q, str) or not q.strip() for q in queries
        ):
            raise ValueError("invalid queries")
        for query in queries:
            query = query.strip()[:self.MAX_QUERY_CHARS]
            if query not in result.queries:
                result.queries.append(query)
            if len(result.queries) >= self.max_queries:
                break
        started = time.perf_counter()
        STAGE.get()["name"] = "search"
        trace("search.start", queries=result.queries, topk=self.topk)
        rows = await self._call(self.search, list(result.queries), self.topk)
        if not isinstance(rows, list):
            raise ValueError("invalid search response")
        merged: dict[str, SourceRef] = {}
        for row in rows[:self.topk * self.max_queries]:
            if not isinstance(row, dict):
                raise ValueError("invalid search item")
            url = canonical_url(row.get("url"))
            if url and url not in merged:
                merged[url] = SourceRef(url, str(row.get("title") or "")[:500],
                                        str(row.get("snippet") or "")[:1500])
        trace("search.results", returned_count=len(rows), unique_valid_count=len(merged),
              candidates=[s.to_dict() for s in merged.values()],
              elapsed_s=round(time.perf_counter() - started, 3))
        if not merged:
            self._empty(result, "no_valid_search_results")
            return
        candidates = {f"S{i}": source for i, source in enumerate(merged.values(), 1)}
        selected = await self._json(SELECT_PROMPT.format(
            limit=self.max_read,
            data=dump({**base, "candidates": {i: s.to_dict() for i, s in candidates.items()}})
        ), result, "select")
        ids = self._ids(selected, candidates)[:self.max_read]
        trace("pages.selected", source_ids=ids, urls=[candidates[sid].url for sid in ids])
        if not ids:
            self._empty(result, "no_pages_selected")
            return
        pages = {}
        raw_pages = {}
        for sid in ids:
            STAGE.get()["name"] = "fetch"
            started = time.perf_counter()
            trace("fetch.start", source_id=sid, url=candidates[sid].url)
            content = await self._call(self.fetch, candidates[sid].url)
            if not isinstance(content, str):
                raise ValueError("invalid page content")
            # 保留有界原文用于摘录核验；筛选阶段不调用LLM或网络。
            raw_pages[sid] = content[:self.MAX_FETCH_CHARS]
            page = select_page_content(raw_pages[sid],
                                       result.sub_question + " " + " ".join(result.queries),
                                       budget=self.MAX_PAGE_CHARS,
                                       scan_limit=self.MAX_FETCH_CHARS)
            trace("fetch.content", source_id=sid, url=candidates[sid].url,
                  received_chars=len(content), supplied_chars=len(page.text),
                  raw_content=raw_pages[sid], supplied_content=page.text,
                  truncated=len(content) > len(page.text), content_preview=page.text[:500],
                  scanned_chars=page.scanned_chars, navigation_chars=page.navigation_chars,
                  selected_spans=page.spans, strategy="lead_and_diverse_visible_terms_v2",
                  elapsed_s=round(time.perf_counter() - started, 3))
            if content.strip():
                result.pages_read += 1
            if page.text.strip():
                pages[sid] = page.text
        if not pages:
            self._empty(result, "no_usable_page_content")
            return
        trace("summary.input", pages=[{"source_id": sid, "url": candidates[sid].url,
                                      "supplied_chars": len(text)} for sid, text in pages.items()])
        summary = await self._json(SUMMARIZE_PROMPT.format(
            data=dump({**base, "pages": pages})), result, "summarize")
        ids = self._ids(summary, pages)
        answer, evidence = summary.get("answer"), summary.get("evidence")
        if not isinstance(answer, str) or not isinstance(evidence, list):
            raise ValueError("invalid summary")
        if not answer.strip() and not ids and not evidence:
            self._empty(result, "summary_declared_insufficient_evidence")
            return
        if not answer.strip() or not ids:
            raise ValueError("answer without sources")
        citations = set(re.findall(r"\[(S[^\]\s]*)\]", answer))
        if citations != set(ids):
            raise ValueError("answer citations mismatch")
        verified = set()
        for item in evidence:
            if not isinstance(item, dict):
                raise ValueError("invalid evidence")
            sid, quote = item.get("source_id"), item.get("quote")
            if not isinstance(sid, str) or sid not in ids or not isinstance(quote, str):
                raise ValueError("unsupported evidence")
            original_quote = resolve_evidence_quote(quote, pages[sid], raw_pages[sid])
            if original_quote is None:
                raise ValueError("unsupported evidence")
            trace("evidence.verified", source_id=sid, quote=quote,
                  original_quote=original_quote, formatting_mapped=quote != original_quote)
            verified.add(sid)
        if verified != set(ids):
            raise ValueError("missing evidence")
        # SourceRef 冻结结构没有 id；按返回 sources 顺序重编号，使 Planner 可定位 [S1] 等。
        renumber = {sid: f"S{i}" for i, sid in enumerate(ids, 1)}
        result.answer = re.sub(r"\[(S[^\]\s]*)\]",
                               lambda match: f"[{renumber[match[1]]}]", answer.strip())
        result.sources = [candidates[sid] for sid in ids]


def make_web_searcher(*, llm: LLM, engine: str = "tavily", topk: int = 3,
                      max_queries: int = 3, max_read: int = 3) -> WebSearcher:
    """按 IFACE v1.7 组装真实检索器；只构造，不调用模型、搜索或抓取。

    llm由调用方提供异步回调；不创建或替换模型。Tavily凭据由后端从
    TAVILY_API_KEY读取。未实现的引擎和非法预算在构造时显式失败。
    """
    if engine != "tavily":
        raise ValueError("make_web_searcher currently supports only engine='tavily'")
    if not callable(llm):
        raise TypeError("make_web_searcher requires an async llm callable")
    for name, value, limit in (("topk", topk, 10), ("max_queries", max_queries, 3),
                               ("max_read", max_read, 10)):
        if type(value) is not int or not 1 <= value <= limit:
            raise ValueError(f"{name} must be an integer between 1 and {limit}")

    from ....backends.tavily_backend import TavilyBackend

    backend = TavilyBackend()
    return WebSearcher(llm=llm, search=backend.search, fetch=backend.fetch,
                       topk=topk, max_queries=max_queries, max_read=max_read)
