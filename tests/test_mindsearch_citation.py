"""citation 校验的离线测试（纯标准库，不需要 langgraph / API）。"""
from __future__ import annotations

import pytest

from lychee_mas.methods.build.mindsearch.citation import (check_citations,
                                                          extract_urls,
                                                          normalize_url)
from lychee_mas.methods.build.mindsearch.dag import TaskDAG


def _dag_with(url: str) -> TaskDAG:
    dag = TaskDAG("q")
    a = dag.add_node("A")
    dag.mark_done(a, "ans", [{"url": url}])
    return dag


def test_normalize_url_strips_scheme_slash_fragment():
    assert normalize_url("HTTP://Example.COM/a/?x=1#frag") == "https://example.com/a?x=1"


def test_extract_urls_dedup_and_strip_punct():
    urls = extract_urls("见 https://a.com/x。另见 https://a.com/x/ 和 https://b.com")
    assert urls == ["https://a.com/x", "https://b.com"]


def test_ok_when_cited_source_matches():
    rep = check_citations("来源：https://a.com/x", _dag_with("https://a.com/x"))
    assert rep.ok and rep.reason == "ok"


def test_unknown_url_rejected():
    rep = check_citations("来源：https://evil.com/fake", _dag_with("https://a.com/x"))
    assert not rep.ok
    assert rep.reason == "unknown_urls"
    assert "https://evil.com/fake" in rep.unknown


def test_no_citation_rejected_by_default_but_optional():
    dag = _dag_with("https://a.com/x")
    assert check_citations("没有引用", dag).reason == "no_citation"
    assert check_citations("没有引用", dag, require_citation=False).ok


def test_no_sources_passes():
    dag = TaskDAG("q")
    a = dag.add_node("A")
    dag.mark_done(a, "ans")                       # 没有 sources（检索失败/为空）
    rep = check_citations("来源：https://whatever.com", dag)
    assert rep.ok and rep.reason == "no_sources"
