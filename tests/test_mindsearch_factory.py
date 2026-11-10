"""真实工厂接线的离线验证：不读本地密钥、不发送HTTP或模型请求。"""
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from lychee_mas.backends import tavily_backend
from lychee_mas.methods.build.mindsearch import WebSearcher, make_web_searcher


async def unused_model(prompt):
    pytest.fail("construction must not call model")


def test_real_backend_is_bound_without_requests(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-test-key")
    monkeypatch.setattr(tavily_backend.TavilyBackend, "_post",
                        lambda *args: pytest.fail("construction must not send requests"))
    s = make_web_searcher(llm=unused_model, topk=2, max_queries=1, max_read=2)
    assert isinstance(s, WebSearcher) and s.llm is unused_model
    assert s.search.__self__ is s.fetch.__self__
    assert isinstance(s.search.__self__, tavily_backend.TavilyBackend)
    assert (s.topk, s.max_queries, s.max_read) == (2, 1, 2)
    assert make_web_searcher(llm=unused_model).search.__self__ is not s.search.__self__


@pytest.mark.parametrize("engine", ["ddg", "wiki", "unknown", "Tavily"])
def test_unsupported_engine_fails_before_key_lookup(monkeypatch, engine):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(ValueError, match="engine"):
        make_web_searcher(llm=unused_model, engine=engine)


@pytest.mark.parametrize("kwargs", [
    {"topk": 0}, {"topk": 11}, {"topk": True}, {"max_queries": 4},
    {"max_queries": "1"}, {"max_read": 0}, {"max_read": 11},
])
def test_invalid_budget_fails_before_key_lookup(monkeypatch, kwargs):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(ValueError, match="must be an integer"):
        make_web_searcher(llm=unused_model, **kwargs)


def test_missing_model_and_key_are_explicit(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(TypeError, match="llm"):
        make_web_searcher(llm=None)
    with pytest.raises(ValueError, match="environment variable"):
        make_web_searcher(llm=unused_model)


def test_e2e_entry_builds_and_runs_real_searcher_with_fake_primitives(monkeypatch):
    calls = []

    class Backend:
        async def search(self, queries, topk):
            calls.append(("search", queries, topk))
            return [{"url": "https://example.org/", "title": "Evidence", "snippet": "text"}]

        async def fetch(self, url):
            calls.append(("fetch", url))
            return "The documented value is 42."

    monkeypatch.setattr(tavily_backend, "TavilyBackend", Backend)
    script = Path(__file__).parents[1] / "scripts" / "run_mindsearch.py"
    spec = importlib.util.spec_from_file_location("mindsearch_factory_entry_test", script)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)

    async def model(prompt):
        if prompt.startswith("Rewrite"):
            return '{"queries":["documented value"]}'
        if prompt.startswith("Select"):
            return '{"source_ids":["S1"]}'
        return json.dumps({"answer": "42 [S1]", "source_ids": ["S1"],
                           "evidence": [{"source_id": "S1", "quote": "documented value is 42"}]})

    args = SimpleNamespace(searcher="real", engine="tavily", topk=2, max_queries=1, max_read=1,
                           timeout=90)
    s = entry.build_searcher(model, args)
    assert s.MODEL_TIMEOUT_S == 90
    assert s.REQUEST_TIMEOUT_S == 35 and s.RUN_TIMEOUT_S == 180
    result = asyncio.run(s.run("What is the documented value?"))
    assert result.is_ok() and result.answer == "42 [S1]"
    assert result.sources[0].url == "https://example.org/"
    assert calls == [("search", ["documented value"], 2), ("fetch", "https://example.org/")]
