"""引用校验（citation check）：答案里的 URL 能否追溯到本次检索的 sources。

职责边界（重要，写进报告）：
- 只做「引用级」校验：答案中出现的 URL 是否出现在 DAG 节点的 sources 里；
- 不做「结论是否被来源支持」的语义校验——那需要 NLI/judge，属已知局限。

设计：
- URL 归一化（去 fragment、去尾斜杠、host 小写、http/https 视同）后再比对；
- 没有任何 sources 时（全部检索失败/空结果）不强制引用，但仍拒绝未知 URL；
- 有 sources 但答案无引用 -> no_citation；引用了未知 URL -> unknown_urls。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit, urlunsplit

_URL_RE = re.compile(r"https?://[^\s<>\"'()\[\]{}，。；、）】]+", re.IGNORECASE)
_TRAIL = ".,;:!?。，；：！？、"


def normalize_url(url: str) -> str:
    """归一化 URL 以便比对：swallow scheme、去 fragment、去尾斜杠、host 小写。"""
    u = (url or "").strip().strip(_TRAIL)
    if not u:
        return ""
    try:
        parts = urlsplit(u)
    except ValueError:
        return u.lower()
    host = (parts.hostname or "").lower()
    if not host:
        return u.lower()
    return urlunsplit(("https", host, parts.path.rstrip("/"), parts.query, ""))


def extract_urls(text: str) -> list[str]:
    """抽取文本中的 URL（去重、归一化、保持出现顺序）。"""
    seen: set[str] = set()
    out: list[str] = []
    for raw in _URL_RE.findall(text or ""):
        norm = normalize_url(raw)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def _source_url(src: Any) -> str:
    if isinstance(src, dict):
        return src.get("url") or ""
    return getattr(src, "url", "") or ""


def dag_sources(dag: Any) -> set[str]:
    """收集 DAG 中所有节点记录过的来源 URL（已归一化）。"""
    known: set[str] = set()
    for node in getattr(dag, "nodes", {}).values():
        for src in getattr(node, "sources", None) or []:
            url = normalize_url(_source_url(src))
            if url:
                known.add(url)
    return known


@dataclass
class CitationReport:
    ok: bool
    reason: str                                  # ok | no_sources | no_citation | unknown_urls
    cited: list[str] = field(default_factory=list)
    known: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)

    def feedback(self) -> str:
        """回灌给 LLM 的修正提示（只描述问题，不给答案）。"""
        if self.reason == "no_citation":
            return "答案没有引用任何来源；请只引用检索结果中出现的来源，或省略来源表述。"
        if self.reason == "unknown_urls":
            return ("答案引用了未检索到的链接："
                    + ", ".join(self.unknown[:3])
                    + "；请只引用检索结果中出现的来源。")
        return ""


def check_citations(answer: str, dag: Any, *, require_citation: bool = True) -> CitationReport:
    """校验答案引用是否可追溯到 DAG 的 sources。

    - 有引用且全部可追溯 -> ok
    - 引用了未知 URL -> unknown_urls（不通过）
    - 有 sources 但完全没引用，且 require_citation -> no_citation（不通过）
    - 一条 sources 都没有且未引用 URL -> no_sources（通过，交给 giveup 路径处理）
    """
    known = dag_sources(dag)
    cited = extract_urls(answer)
    if not known and not cited:
        return CitationReport(ok=True, reason="no_sources", cited=cited, known=[])
    known_lower = {u.lower() for u in known}
    unknown = [u for u in cited if u not in known and u.lower() not in known_lower]
    if unknown:
        return CitationReport(ok=False, reason="unknown_urls", cited=cited,
                              known=sorted(known), unknown=unknown)
    if require_citation and not cited:
        return CitationReport(ok=False, reason="no_citation", cited=cited,
                              known=sorted(known))
    return CitationReport(ok=True, reason="ok", cited=cited, known=sorted(known))
