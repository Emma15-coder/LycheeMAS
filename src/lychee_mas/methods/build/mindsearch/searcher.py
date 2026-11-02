"""WebSearcher —— 分层检索流水（B 负责实现，签名与语义在此冻结）。

四步：① 多 query 改写 ② 按 URL 合并搜索结果 ③ 选页读正文 ④ 总结回填。

三条冻结约定（改动需 A/B 双方同意并单独提交 refactor）：
1. 失败不抛异常：网络/抓取失败返回 status="error" + error 文本，run() 内部最多重试 1 次；
   由 planner 决定是重规划还是 mark_failed——本层不假装成功，也不抛穿整个 run。
2. 空结果：检索到了但没有可用内容 -> status="empty"、answer=""。
3. context 语义：固定为「父节点 answer + 分隔线 + 根问题」，不含兄弟节点与全部历史。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

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

    async def run(self, sub_question: str, context: str = "") -> SearcherResult:
        raise NotImplementedError(
            "WebSearcher.run 待实现（B）：① 多 query 改写 ② 按 URL 合并 ③ 选页读正文 "
            "④ 总结回填；遵守上限 max_queries/topk/max_read，token 写回 SearcherResult")
