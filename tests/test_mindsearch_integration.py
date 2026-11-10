"""真实Planner + WebSearcher + StateGraph离线联调，只替换模型与搜索/抓取原语。"""
import asyncio
import importlib.util
import json
from collections import Counter, deque
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("langgraph")

from lychee_mas.backends import tavily_backend  # noqa: E402
from lychee_mas.methods.build.mindsearch import (  # noqa: E402
    MindSearchState,
    planner_node_factory,
)
from lychee_mas.plugins import build_langgraph  # noqa: E402


@pytest.fixture
def entry():
    path = Path(__file__).parents[1] / "scripts" / "run_mindsearch.py"
    spec = importlib.util.spec_from_file_location("mindsearch_integration_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expand(*queries):
    return {"nodes": [{"query": q, "parents": []} for q in queries]}


def finish(answer="Exampleland", url="https://evidence.example/country", parents=None):
    return {"finish": f"Answer: {answer}\nSource: {url}" if url else f"Answer: {answer}",
            "response_parents": parents or ["n0"]}


class Pipeline:
    def __init__(self, plans, documents, *, fault=None):
        self.plans = deque(plans)
        self.documents = documents
        self.fault = fault
        self.responses = []
        self.prompts = []
        self.events = []
        self.contexts = {}
        self.stages = Counter()
        self.active = 0
        self.peak = 0

    async def llm(self, prompt):
        await asyncio.sleep(0)
        self.prompts.append(prompt)
        if "Current task graph:" in prompt:
            self.stages["planner"] += 1
            assert self.plans, "unplanned extra Planner call"
            payload = self.plans.popleft()
        else:
            data = json.loads(prompt.split("Input JSON:\n", 1)[1])
            q = data["sub_question"]
            self.contexts.setdefault(q, []).append(data["context"])
            if prompt.startswith("Rewrite"):
                self.stages["rewrite"] += 1
                payload = {"queries": [q]}
            elif prompt.startswith("Select"):
                self.stages["select"] += 1
                payload = {"source_ids": ["S1"]}
            else:
                self.stages["summary"] += 1
                _, body, answer = self.documents[q]
                assert body in data["pages"]["S1"]
                if self.fault == "insufficient":
                    payload = {"answer": "", "source_ids": [], "evidence": []}
                elif self.fault == "bad_source":
                    payload = {"answer": "made up [S99]", "source_ids": ["S99"], "evidence": []}
                else:
                    payload = {"answer": f"{answer} [S1]", "source_ids": ["S1"],
                               "evidence": [{"source_id": "S1", "quote": body}]}
        # 每次调用的usage不同，检查最终是否遗漏或重复记账。
        response = SimpleNamespace(text=json.dumps(payload), n_prompt_pos=10 + len(self.responses),
                                   n_gen_tokens=2, finish_reason="stop")
        self.responses.append(response)
        return response

    def backend(self):
        pipeline = self

        class Backend:
            async def search(self, queries, topk):
                assert len(queries) == 1 and topk == 1
                q = queries[0]
                pipeline.events.append(("search.start", q))
                pipeline.active += 1
                pipeline.peak = max(pipeline.peak, pipeline.active)
                try:
                    await asyncio.sleep(0)
                    if pipeline.fault in ("search_error", f"search_error:{q}"):
                        raise OSError("Bearer fake-secret")
                    url, body, _ = pipeline.documents[q]
                    return [{"url": url, "title": q, "snippet": body}]
                finally:
                    pipeline.active -= 1
                    pipeline.events.append(("search.end", q))

            async def fetch(self, url):
                pipeline.events.append(("fetch", url))
                if pipeline.fault == "fetch_timeout":
                    raise asyncio.TimeoutError("fake secret timeout detail")
                for candidate_url, body, _ in pipeline.documents.values():
                    if url == candidate_url:
                        return "" if pipeline.fault == "empty_body" else body
                pytest.fail("unknown fetch URL")

        return Backend

    def assert_usage(self, result):
        assert result["prompt_tokens"] == sum(r.n_prompt_pos for r in self.responses)
        assert result["completion_tokens"] == sum(r.n_gen_tokens for r in self.responses)
        for prefix, planner in (("planner", True), ("searcher", False)):
            responses = [r for prompt, r in zip(self.prompts, self.responses)
                         if ("Current task graph:" in prompt) == planner]
            assert result[f"{prefix}_prompt_tokens"] == sum(r.n_prompt_pos for r in responses)
            assert result[f"{prefix}_completion_tokens"] == sum(r.n_gen_tokens for r in responses)
        assert result["queries_count"] == sum(e[0] == "search.start" for e in self.events)
        expected_pages = (0 if self.fault in ("empty_body", "fetch_timeout") else
                          sum(e[0] == "fetch" for e in self.events))
        assert result["pages_read"] == expected_pages
        assert not self.plans


def run_pipeline(entry, monkeypatch, pipeline, question="ROOT", **planner_options):
    monkeypatch.setattr(tavily_backend, "TavilyBackend", pipeline.backend())
    args = SimpleNamespace(searcher="real", engine="tavily", topk=1, max_queries=1, max_read=1)
    searcher = entry.build_searcher(pipeline.llm, args)
    factory = planner_node_factory(searcher=searcher, llm=pipeline.llm,
                                   **{"max_turns": 5, **planner_options})
    app = build_langgraph(method="mindsearch", node_factory=factory,
                          state_schema=MindSearchState).compile()
    result = asyncio.run(entry.run_mindsearch(app, question))
    pipeline.assert_usage(result)
    return result


def test_two_hop_context_sources_cost_and_artifacts(entry, monkeypatch, tmp_path):
    documents = {
        "birthplace": ("https://evidence.example/birth", "Person was born in Harbor City.",
                       "Harbor City"),
        "sibling": ("https://evidence.example/sibling", "SIBLING_ONLY evidence.", "SIBLING_ONLY"),
        "country": ("https://evidence.example/country", "Harbor City is in Exampleland.",
                    "Exampleland"),
    }
    pipeline = Pipeline([expand("birthplace", "sibling"),
                         {"nodes": [{"query": "country", "parents": ["n0"]}]},
                         finish(parents=["n2"])], documents)
    result = run_pipeline(entry, monkeypatch, pipeline)
    assert result["answer"] == "Exampleland" and result["stop_reason"] == "finish"
    assert result["format_ok"] and result["citation_ok"] and not result["truncated"]
    nodes = {n["qid"]: n for n in result["dag"]["nodes"]}
    assert nodes["n2"]["parents"] == ["n0"]
    final = next(n for n in nodes.values() if n["query"] == "[FINAL RESPONSE]")
    assert final["parents"] == ["n2"]
    assert final["sources"][0]["url"] == documents["country"][0]
    for context in pipeline.contexts["country"]:
        assert "[n0] Harbor City [S1]" in context
        assert "\n---\nRoot question: ROOT" in context and "SIBLING_ONLY" not in context
    assert pipeline.peak == 2
    assert pipeline.stages == {"planner": 3, "rewrite": 3, "select": 3, "summary": 3}
    entry.write_case_artifacts(tmp_path, "0", result["dag"])
    entry.write_sources(tmp_path, "0", result["dag"])
    assert json.loads((tmp_path / "dag/0.json").read_text()) == result["dag"]
    assert (tmp_path / "dag/0.mmd").read_text()
    sources = [json.loads(line) for line in (tmp_path / "sources.jsonl").read_text().splitlines()]
    assert all(s["case_id"] == "0" for s in sources)
    assert any(s["qid"] == final["qid"] and s["url"] == documents["country"][0] for s in sources)


@pytest.mark.parametrize("fault,status", [("empty_body", "done"), ("insufficient", "done"),
                                         ("search_error", "failed"), ("bad_source", "failed"),
                                         ("fetch_timeout", "failed")])
def test_empty_and_error_propagate_without_sources(entry, monkeypatch, fault, status):
    pipeline = Pipeline([expand("subquestion"), finish("Cannot confirm", url="")],
                        {"subquestion": ("https://evidence.example/item", "Partial evidence.",
                                         "partial")}, fault=fault)
    result = run_pipeline(entry, monkeypatch, pipeline, max_fail_streak=1)
    assert result["stop_reason"] == "giveup" and result["truncated"]
    assert "[UNRESOLVED]" in result["answer_raw"]
    node = result["dag"]["nodes"][0]
    assert node["status"] == status and not node["sources"]
    assert "fake-secret" not in json.dumps(result)
    if fault == "search_error":
        assert pipeline.stages["select"] == pipeline.stages["summary"] == 0
    elif fault == "empty_body":
        assert pipeline.stages["summary"] == 0


def test_unknown_final_url_rejected_then_corrected(entry, monkeypatch):
    good = "https://evidence.example/country"
    pipeline = Pipeline([expand("country"), finish(url="https://unknown.example/"), finish()],
                        {"country": (good, "The country is Exampleland.", "Exampleland")})
    result = run_pipeline(entry, monkeypatch, pipeline)
    assert result["citation_ok"] and "unknown.example" not in result["answer_raw"]
    assert "unknown.example" in pipeline.prompts[-1]  # 回灌而非重新检索
    assert pipeline.stages["rewrite"] == pipeline.stages["summary"] == 1


def test_turn_budget_stops_before_unexecuted_search(entry, monkeypatch):
    pipeline = Pipeline([expand("not searched"), finish("Cannot confirm", url="")], {})
    result = run_pipeline(entry, monkeypatch, pipeline, max_turns=1)
    assert result["stop_reason"] == "max_turns" and result["truncated"]
    assert result["turns"] == 1 and "[UNRESOLVED]" in result["answer_raw"]
    assert not pipeline.events and pipeline.stages == {"planner": 2}


def test_no_source_still_rejects_fabricated_final_url(entry, monkeypatch):
    pipeline = Pipeline([expand("subquestion"), finish("Cannot confirm", "https://unknown.example/")],
                        {"subquestion": ("https://evidence.example/item", "partial", "partial")},
                        fault="empty_body")
    result = run_pipeline(entry, monkeypatch, pipeline, max_fail_streak=1)
    assert result["stop_reason"] == "giveup"
    assert not result["citation_ok"]
    assert "https://unknown.example" in result["citation_unknown"]


def test_retrieval_concurrency_budget_is_enforced(entry, monkeypatch):
    pipeline = Pipeline([expand("left", "right"), finish()], {
        "left": ("https://evidence.example/country", "Value is Exampleland.", "Exampleland"),
        "right": ("https://evidence.example/other", "Other evidence.", "other"),
    })
    result = run_pipeline(entry, monkeypatch, pipeline, max_concurrency=1)
    assert result["stop_reason"] == "finish" and result["citation_ok"]
    assert pipeline.peak == 1 and pipeline.stages["summary"] == 2


def test_failed_parent_blocks_child_and_alternative_can_finish(entry, monkeypatch):
    pipeline = Pipeline([
        expand("bad"),
        {"nodes": [{"query": "blocked", "parents": ["n0"]},
                   {"query": "alternative", "parents": []}]},
        finish(parents=["n2"]),
    ], {
        "bad": ("https://evidence.example/bad", "bad evidence", "bad"),
        "blocked": ("https://evidence.example/blocked", "must not be read", "blocked"),
        "alternative": ("https://evidence.example/country", "Value is Exampleland.", "Exampleland"),
    }, fault="search_error:bad")
    result = run_pipeline(entry, monkeypatch, pipeline)
    nodes = {n["qid"]: n for n in result["dag"]["nodes"]}
    assert result["stop_reason"] == "finish" and result["citation_ok"]
    assert nodes["n0"]["status"] == "failed" and nodes["n1"]["status"] == "pending"
    assert nodes["n2"]["status"] == "done" and "blocked" not in pipeline.contexts
    assert pipeline.stages["rewrite"] == 2 and pipeline.stages["summary"] == 1


def test_failed_attempts_and_repeated_pages_accumulate_across_retrieval_rounds():
    from lychee_mas.methods.build.mindsearch import SearcherResult, TaskDAG
    from lychee_mas.methods.build.mindsearch.planner import _retrieve

    dag = TaskDAG('root')
    first = dag.add_node('first')
    second = dag.add_node('second')
    tok = {'p': 10, 'c': 2, 'planner_prompt_tokens': 10, 'planner_completion_tokens': 2}

    class FailedSearcher:
        async def run(self, question, context):
            return SearcherResult(sub_question=question, status='error',
                                  queries=['same-query', 'another-query'], pages_read=1,
                                  prompt_tokens=7, completion_tokens=3, error='summary failed')

    async def scenario():
        for qid in (first, second):
            assert await _retrieve([qid], dag, FailedSearcher(), tok, 1) == (0, 0, 1)

    asyncio.run(scenario())
    assert tok['queries_count'] == 4 and tok['pages_read'] == 2
    assert tok['searcher_prompt_tokens'] == 14 and tok['searcher_completion_tokens'] == 6
    assert tok['p'] == tok['planner_prompt_tokens'] + tok['searcher_prompt_tokens'] == 24
    assert tok['c'] == tok['planner_completion_tokens'] + tok['searcher_completion_tokens'] == 8
