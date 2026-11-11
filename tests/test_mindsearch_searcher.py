import asyncio
import json
from types import SimpleNamespace

import pytest
from lychee_mas.methods.build.mindsearch.searcher import WebSearcher

URL = "https://example.org/a"
TEXT = "Marie Curie was born in Warsaw. Warsaw is in Poland."


def summary(**changes):
    data = {"answer": "华沙，波兰。[S1]", "source_ids": ["S1"],
            "evidence": [{"source_id": "S1", "quote": "born in Warsaw"}]}
    return {**data, **changes}


class Fake:
    def __init__(self, outputs=None, rows=None, body=TEXT):
        self.outputs = outputs or [{"queries": ["birthplace"]},
                                   {"source_ids": ["S1"]}, summary()]
        self.rows = rows if rows is not None else [
            {"url": URL, "title": "Curie", "snippet": "birthplace"}]
        self.body = body
        self.prompts = []
        self.searches = []
        self.fetches = []

    async def llm(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(text=json.dumps(self.outputs[len(self.prompts)-1]),
                               n_prompt_pos=10, n_gen_tokens=5)

    async def search(self, queries, topk):
        self.searches.append((queries, topk))
        return self.rows

    async def fetch(self, url):
        self.fetches.append(url)
        return self.body

    def searcher(self, **kw):
        return WebSearcher(self.llm, self.search, self.fetch, **kw)


def test_normal_and_context_and_accounting():
    fake = Fake()
    result = asyncio.run(fake.searcher().run("birthplace?", "parent\n---\nroot"))
    assert result.is_ok() and result.answer == "华沙，波兰。[S1]"
    assert result.sources[0].to_dict() == {"url": URL, "title": "Curie",
                                           "snippet": "birthplace"}
    assert result.queries == ["birthplace"] and result.pages_read == 1
    assert (result.prompt_tokens, result.completion_tokens) == (30, 15)
    assert all('parent\\n---\\nroot' in p for p in fake.prompts)


def test_dedupe_invalid_urls_and_budgets():
    fake = Fake(outputs=[{"queries": [" a ", "a", "b", "c"]},
                         {"source_ids": ["S1", "S2"]}, summary()], rows=[
        {"url": "https://EXAMPLE.org:443/a#x"}, {"url": URL},
        {"url": "javascript:bad"}, {"url": "https://example.org/b"}])
    result = asyncio.run(fake.searcher(max_queries=2, topk=3, max_read=1).run("q"))
    assert result.is_ok() and result.queries == ["a", "b"]
    assert fake.fetches == [URL]
    assert fake.prompts[1].count('"url":') == 2


@pytest.mark.parametrize("rows,body", [([], TEXT), ([{"url": "file:///x"}], TEXT),
                                       ([{"url": URL}], "")])
def test_empty_never_answers_from_snippets(rows, body):
    fake = Fake(rows=rows, body=body)
    result = asyncio.run(fake.searcher().run("q"))
    assert result.status == "empty" and not result.answer and not result.sources
    assert len(fake.prompts) < 3


@pytest.mark.parametrize("stage", ["llm", "search", "fetch"])
def test_failure_redacted_no_retry_and_keeps_usage(stage):
    fake = Fake()
    searcher = fake.searcher()
    calls = []

    async def fail(*args):
        calls.append(1)
        raise OSError("Bearer secret-key")

    setattr(searcher, stage, fail)
    result = asyncio.run(searcher.run("q"))
    assert result.status == "error" and "secret" not in result.error
    assert not result.answer and len(calls) == 1
    assert result.prompt_tokens == {"llm": 0, "search": 10, "fetch": 20}[stage]


@pytest.mark.parametrize("bad", [
    summary(source_ids=["S99"]), summary(answer="false [S99]"),
    summary(evidence=[{"source_id": "S1", "quote": "Paris"}]),
    summary(evidence=[]), summary(source_ids=["S1", "S1"]),
    summary(answer="uncited"), summary(evidence=[{"source_id": "S2", "quote": "Warsaw"}]),
])
def test_invalid_or_fabricated_citations(bad):
    fake = Fake(outputs=[{"queries": ["a"]}, {"source_ids": ["S1"]}, bad])
    result = asyncio.run(fake.searcher().run("q"))
    assert result.status == "error" and not result.sources and not result.answer
    assert result.pages_read == 1 and result.completion_tokens == 15


def test_insufficient_evidence_empty():
    fake = Fake(outputs=[{"queries": ["a"]}, {"source_ids": ["S1"]},
                         {"answer": "", "source_ids": [], "evidence": []}])
    assert asyncio.run(fake.searcher().run("q")).status == "empty"


def test_page_truncation_prevents_citing_unseen_text():
    fake = Fake(body="x" * 12000 + "born in Warsaw")
    result = asyncio.run(fake.searcher().run("q"))
    assert result.status == "error"
    assert "born in Warsaw" not in fake.prompts[-1]


def test_timeout_and_cancellation():
    async def scenario():
        fake = Fake()
        s = fake.searcher()
        s.REQUEST_TIMEOUT_S = 0.001

        async def slow(*args):
            await asyncio.sleep(10)

        s.search = slow
        assert (await s.run("q")).status == "error"
        task = asyncio.create_task(s.run("q"))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


@pytest.mark.parametrize("kw,question,context", [
    ({"max_read": 0}, "q", ""), ({}, "", ""), ({}, "q", "x" * 8001)])
def test_invalid_budget_no_calls(kw, question, context):
    fake = Fake()
    assert asyncio.run(fake.searcher(**kw).run(question, context)).status == "error"
    assert not fake.prompts


def test_concurrent_results_do_not_share_state():
    async def scenario():
        async def llm(prompt):
            if prompt.startswith("Rewrite"):
                return '{"queries":["x"]}'
            if prompt.startswith("Select"):
                return '{"source_ids":["S1"]}'
            return json.dumps(summary())

        fake = Fake()
        s = WebSearcher(llm, fake.search, fake.fetch)
        a, b = await asyncio.gather(s.run("a"), s.run("b"))
        assert a.is_ok() and b.is_ok() and a.queries is not b.queries
        assert a.sources is not b.sources and a.pages_read == b.pages_read == 1

    asyncio.run(scenario())


def test_noncontiguous_sources_are_renumbered_for_planner():
    fake = Fake(outputs=[{"queries": ["q"]}, {"source_ids": ["S2"]},
                         summary(answer="华沙 [S2]", source_ids=["S2"],
                                 evidence=[{"source_id": "S2", "quote": "Warsaw"}])],
                rows=[{"url": URL}, {"url": "https://example.org/b"}])
    result = asyncio.run(fake.searcher().run("q"))
    assert result.is_ok() and result.answer == "华沙 [S1]"
    assert result.sources[0].url == "https://example.org/b"


def test_failed_second_page_preserves_successful_page_count():
    fake = Fake(outputs=[{"queries": ["q"]}, {"source_ids": ["S1", "S2"]}],
                rows=[{"url": URL}, {"url": "https://example.org/b"}])
    s = fake.searcher()

    async def fetch(url):
        if url.endswith("/b"):
            raise OSError("secret")
        return TEXT

    s.fetch = fetch
    result = asyncio.run(s.run("q"))
    assert result.status == "error" and result.pages_read == 1 and not result.sources


def test_invalid_json_is_error_and_accounted():
    fake = Fake()
    s = fake.searcher()

    async def llm(prompt):
        return SimpleNamespace(text="```json nope", n_prompt_pos=3, n_gen_tokens=2)

    s.llm = llm
    result = asyncio.run(s.run("q"))
    assert result.status == "error" and result.prompt_tokens == 3
    assert not fake.searches


def test_whole_run_timeout():
    async def scenario():
        fake = Fake()
        s = fake.searcher()
        s.RUN_TIMEOUT_S = 0.001

        async def slow(*args):
            await asyncio.sleep(1)

        s.llm = slow
        result = await s.run("q")
        assert result.status == "error" and "TimeoutError" in result.error

    asyncio.run(scenario())


def trace_events(caplog):
    return [json.loads(record.getMessage()) for record in caplog.records
            if record.name == "lychee_mas.mindsearch.trace"]


def test_replay_pages_read_one_summary_empty_branch(caplog):
    """复现截图的状态组合，不假设旧运行未记录的URL、正文或finish_reason。"""
    caplog.set_level("INFO", logger="lychee_mas.mindsearch.trace")
    fake = Fake(outputs=[{"queries": ["a"]}, {"source_ids": ["S1"]},
                         {"answer": "", "source_ids": [], "evidence": []}])
    result = asyncio.run(fake.searcher().run("q"))
    assert result.status == "empty" and result.pages_read == 1
    events = trace_events(caplog)
    empty = next(e for e in events if e["event"] == "searcher.empty")
    assert empty["reason"] == "summary_declared_insufficient_evidence"
    responses = [e for e in events if e["event"] == "model.response"]
    assert [e["stage"] for e in responses] == ["rewrite", "select", "summarize"]
    assert json.loads(responses[-1]["text"])["answer"] == ""
    assert responses[-1]["finish_reason"] is None  # 旧/裸回调无法提供时明确为null
    assert len({e["run_id"] for e in events}) == 1


@pytest.mark.parametrize("case,reason", [
    ("search", "no_valid_search_results"),
    ("select", "no_pages_selected"),
    ("fetch", "no_usable_page_content"),
])
def test_other_empty_branches_have_distinct_reasons(caplog, case, reason):
    caplog.set_level("INFO", logger="lychee_mas.mindsearch.trace")
    fake = Fake(rows=[] if case == "search" else None,
                body=" " if case == "fetch" else TEXT)
    if case == "select":
        fake.outputs[1] = {"source_ids": []}
    result = asyncio.run(fake.searcher().run("q"))
    assert result.status == "empty" and result.pages_read == 0
    assert next(e for e in trace_events(caplog)
                if e["event"] == "searcher.empty")["reason"] == reason


def test_trace_counts_selected_url_and_truncation(caplog):
    caplog.set_level("INFO", logger="lychee_mas.mindsearch.trace")
    fake = Fake(body=TEXT + "x" * 13000)
    result = asyncio.run(fake.searcher().run("q"))
    assert result.is_ok()
    events = trace_events(caplog)
    search = next(e for e in events if e["event"] == "search.results")
    assert search["returned_count"] == search["unique_valid_count"] == 1
    selected = next(e for e in events if e["event"] == "pages.selected")
    assert selected["urls"] == [URL]
    fetch = next(e for e in events if e["event"] == "fetch.content")
    assert fetch["received_chars"] == len(fake.body)
    assert 0 < fetch["supplied_chars"] <= 12000 and fetch["truncated"] is True
    supplied = next(e for e in events if e["event"] == "summary.input")
    assert supplied["pages"][0]["supplied_chars"] == fetch["supplied_chars"]


@pytest.mark.parametrize("reason", ["length", "content_filter"])
def test_incomplete_even_valid_empty_json_is_error(caplog, reason):
    caplog.set_level("INFO", logger="lychee_mas.mindsearch.trace")
    fake = Fake()
    s = fake.searcher()
    original = fake.llm

    async def llm(prompt):
        response = await original(prompt)
        response.finish_reason = "stop"
        if len(fake.prompts) == 3:
            response.text = '{"answer":"","source_ids":[],"evidence":[]}'
            response.finish_reason = reason
        return response

    s.llm = llm
    result = asyncio.run(s.run("q"))
    assert result.status == "error" and result.pages_read == 1
    assert result.prompt_tokens == 30 and result.completion_tokens == 15
    event = next(e for e in trace_events(caplog) if e["event"] == "model.incomplete")
    assert event["stage"] == "summarize" and event["finish_reason"] == reason
    assert not any(e["event"] == "searcher.empty" for e in trace_events(caplog))


def test_trace_redacts_response_preview_url_and_environment_keys(monkeypatch, caplog):
    caplog.set_level("INFO", logger="lychee_mas.mindsearch.trace")
    monkeypatch.setenv("SILICONFLOW_API_KEY", "model-private-123")
    monkeypatch.setenv("TAVILY_API_KEY", "extract-private-456")
    monkeypatch.setenv("CUSTOM_API_KEY", 'custom-"quoted"-secret')
    fake = Fake(body=TEXT + ' model-private-123 extract-private-456 custom-"quoted"-secret')
    fake.rows[0]["url"] = "https://x.org/?api_key=unknown-url-secret"
    fake.outputs[0] = {"queries": ["model-private-123 tvly-testkey Bearer sk-fakekey"]}
    result = asyncio.run(fake.searcher().run("q"))
    assert result.is_ok()
    logs = "\n".join(record.getMessage() for record in caplog.records)
    for secret in ("model-private-123", "extract-private-456", "custom-",
                   "tvly-testkey", "sk-fakekey", "unknown-url-secret"):
        assert secret not in logs
    assert "[REDACTED]" in logs
    trace_events(caplog)  # 脱敏后仍是合法JSON，防止引号/换行破坏结构


@pytest.mark.parametrize("fault,stage,reason", [
    ("quote", "summarize", "unsupported_evidence"),
    ("ids", "select", "invalid_source_IDs"),
    ("json", "rewrite", "invalid_json"),
    ("timeout", "search", "timeout"),
    ("external", "search", "external_or_validation_error"),
])
def test_diagnostic_reason_is_specific_and_external_errors_stay_private(fault, stage, reason):
    fake = Fake()
    if fault == "quote":
        fake.outputs[2] = summary(evidence=[{"source_id": "S1", "quote": "invented"}])
    elif fault == "ids":
        fake.outputs[1] = {"source_ids": ["S99"]}
    searcher = fake.searcher()
    if fault == "json":
        async def invalid(prompt):
            return "not json"
        searcher.llm = invalid
    if fault in ("timeout", "external"):
        async def failed(queries, topk):
            if fault == "timeout":
                raise TimeoutError("private-provider-message")
            raise ValueError("private-provider-message")
        searcher.search = failed
    result = asyncio.run(searcher.run("q"))
    assert result.status == "error"
    assert f"stage={stage}; reason={reason}" in result.error
    assert "private-provider-message" not in result.error


def test_trace_file_preserves_replay_material_and_restores_logger(tmp_path, monkeypatch):
    from lychee_mas.backends.searcher_trace import LOGGER, redact_fields, trace_file

    monkeypatch.setenv("TEST_API_KEY", "private-key-for-test")
    fake = Fake(body=TEXT + " private-key-for-test")
    path = tmp_path / "trace.jsonl"
    previous = LOGGER.level
    handlers = list(LOGGER.handlers)
    with trace_file(path, enabled=True):
        assert asyncio.run(fake.searcher().run("q")).is_ok()
    assert LOGGER.level == previous and LOGGER.handlers == handlers
    assert path.stat().st_mode & 0o777 == 0o600
    text = path.read_text()
    assert "private-key-for-test" not in text
    events = [json.loads(line) for line in text.splitlines()]
    starts = [e for e in events if e["event"] == "model.start"]
    assert [e["prompt"] for e in starts] == redact_fields(fake.prompts)
    page = next(e for e in events if e["event"] == "fetch.content")
    assert TEXT in page["raw_content"] and TEXT in page["supplied_content"]
    assert len({e["run_id"] for e in events}) == 1


def test_captured_real_quotes_with_markdown_and_nonbreaking_spaces():
    from pathlib import Path

    fixture = Path(__file__).parent / 'fixtures/mindsearch_quote_formatting.json'
    cases = json.loads(fixture.read_text())
    for case in cases:
        quote = case['summary']['evidence'][0]['quote']
        assert quote not in case['excerpt']  # 原始实现的失败条件
        fake = Fake(outputs=[{'queries': ['test']}, {'source_ids': ['S1']}, case['summary']],
                    body=case['excerpt'])
        result = asyncio.run(fake.searcher().run('test'))
        assert result.is_ok(), result.error
        # 有格式噪声仍不能接受改写事实或替换实体。
        fake.outputs[2] = summary(evidence=[{'source_id': 'S1', 'quote': quote + ' invented'}])
        fake.prompts.clear()
        assert asyncio.run(fake.searcher().run('test')).status == 'error'


@pytest.mark.parametrize('quote,supplied,raw', [
    ('Darvel, Scotland', '[Darvel](https://example.org), England',
     '[Darvel](https://example.org), England'),
    ('born in Scotland', 'born in\n\nScotland', 'born in another place. Scotland'),
    ('born in Scotland', 'born in Scotland', 'born in another place. Scotland'),
    ('1812', '1813', '1813'),
])
def test_formatting_mapping_never_accepts_changed_facts_or_stitched_text(quote, supplied, raw):
    from lychee_mas.methods.build.mindsearch.page_content import resolve_evidence_quote

    assert resolve_evidence_quote(quote, supplied, raw) is None


def test_model_timeout_is_separate_from_search_timeout():
    async def scenario():
        fake = Fake()
        searcher = fake.searcher()
        searcher.REQUEST_TIMEOUT_S = 0.001
        searcher.MODEL_TIMEOUT_S = 1
        model = fake.llm

        async def delayed(prompt):
            await asyncio.sleep(0.01)
            return await model(prompt)

        searcher.llm = delayed
        assert (await searcher.run('q')).is_ok()
        fake.prompts.clear()
        searcher.MODEL_TIMEOUT_S = 0.001
        result = await searcher.run('q')
        assert result.status == 'error' and 'stage=rewrite; reason=timeout' in result.error

    asyncio.run(scenario())


@pytest.mark.parametrize('stage,reason,status', [
    ('search', 'http_error', 429),
    ('search', 'connection_error', None),
    ('search', 'invalid_json', None),
    ('fetch', 'extract_failed', None),
    ('fetch', 'network_timeout', None),
])
def test_backend_safe_metadata_reaches_searcher_result_and_trace(caplog, stage, reason, status):
    from lychee_mas.backends.search_errors import SearchBackendError

    caplog.set_level('INFO', logger='lychee_mas.mindsearch.trace')
    fake = Fake()
    searcher = fake.searcher()

    async def fail(*args):
        raise SearchBackendError(reason, http_status=status)

    setattr(searcher, stage, fail)
    result = asyncio.run(searcher.run('q'))
    assert result.status == 'error' and result.answer == '' and result.sources == []
    assert f'stage={stage}; reason={reason}' in result.error
    event = next(e for e in trace_events(caplog) if e['event'] == 'searcher.error')
    assert event['stage'] == stage and event['reason'] == reason
    assert event['http_status'] == status
    if status:
        assert f'http_status={status}' in result.error


@pytest.mark.parametrize('text,quote', [
    ('Founded as **City Bank of New York**, in 1812.',
     'Founded as City Bank of New York, in 1812.'),
    ('Founded as __City Bank of New York__, in 1812.',
     'Founded as City Bank of New York, in 1812.'),
    ('Born in **[Darvel](https://example.org/darvel)**, Scotland.',
     'Born in Darvel, Scotland.'),
    ('**Born** in\u00a0Scotland.', 'Born in Scotland.'),
])
def test_bold_quote_mapping_returns_contiguous_original(text, quote):
    from lychee_mas.methods.build.mindsearch.page_content import resolve_evidence_quote

    original = resolve_evidence_quote(quote, text, text)
    assert original is not None and original in text
    assert '**' in original or '__' in original


@pytest.mark.parametrize('text,quote', [
    ('Founded in **1812**.', 'Founded in 1813.'),
    ('Born in **Scotland**.', 'Born in England.'),
    ('Founded as **City Bank.', 'Founded as City Bank.'),
    ('Count = 2 * 3.', 'Count = 2  3.'),
    ('code: `a ** b ** c`', 'code: `a b c`'),
    (r'Literal \**word**.', 'Literal word.'),
    ('identifier a__b__c', 'identifier abc'),
    ('Founded in **1812**.\n\nBorn in Scotland.', 'Founded in 1812. Born in Scotland.'),
])
def test_bold_mapping_preserves_facts_literals_and_boundaries(text, quote):
    from lychee_mas.methods.build.mindsearch.page_content import resolve_evidence_quote

    assert resolve_evidence_quote(quote, text, text) is None
