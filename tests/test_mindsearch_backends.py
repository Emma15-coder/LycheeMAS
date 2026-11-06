import asyncio
import io
import json
import threading
import urllib.error

import pytest
from lychee_mas.backends.async_chat import AsyncChat
from lychee_mas.backends.tavily_backend import TavilyBackend


class Response(io.BytesIO):
    status = 200


class Transport:
    def __init__(self, payloads):
        self.payloads = iter(payloads)
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        payload = next(self.payloads)
        if isinstance(payload, Exception):
            raise payload
        return Response(json.dumps(payload).encode())


def backend(monkeypatch, payloads):
    monkeypatch.setenv("TAVILY_API_KEY", "test-secret")
    b = TavilyBackend()
    b._opener = Transport(payloads)
    return b


def test_tavily_requests_and_mapping(monkeypatch):
    b = backend(monkeypatch, [
        {"results": [{"url": "https://x.org/", "title": "title", "content": "snippet"}]},
        {"results": [{"url": "https://x.org/", "raw_content": "body"}], "failed_results": []}])

    async def scenario():
        rows = await b.search(["q", "q"], 3)
        assert rows == [{"url": "https://x.org/", "title": "title", "snippet": "snippet"}]
        assert await b.fetch(rows[0]["url"]) == "body"

    asyncio.run(scenario())
    requests = [json.loads(r.data) for r, _ in b._opener.calls]
    assert requests[0] == {"query": "q", "search_depth": "basic", "max_results": 3,
                           "include_answer": False, "include_raw_content": False,
                           "auto_parameters": False}
    assert requests[1] == {"urls": ["https://x.org/"], "extract_depth": "basic",
                           "format": "markdown"}
    assert all(timeout == 20 for _, timeout in b._opener.calls)


@pytest.mark.parametrize("payload", [
    urllib.error.HTTPError("https://x", 401, "secret", {}, None),
    OSError("test-secret"), {"detail": "test-secret"}, {"results": "bad"}])
def test_tavily_errors_redacted_no_retries(monkeypatch, payload):
    b = backend(monkeypatch, [payload])
    with pytest.raises(RuntimeError) as exc:
        asyncio.run(b.search(["q"], 3))
    assert "test-secret" not in str(exc.value) and len(b._opener.calls) == 1


@pytest.mark.parametrize("payload", [
    {"results": [], "failed_results": [{"error": "test-secret"}]},
    {"results": [{"url": "https://other.org/", "raw_content": "body"}]},
])
def test_extract_failed_or_mismatched_url(monkeypatch, payload):
    b = backend(monkeypatch, [payload])
    with pytest.raises(RuntimeError):
        asyncio.run(b.fetch("https://x.org/"))


def test_extract_truncation_and_empty(monkeypatch):
    b = backend(monkeypatch, [
        {"results": [{"url": "https://x.org/", "raw_content": "x" * 13000}]},
        {"results": []}])
    assert len(asyncio.run(b.fetch("https://x.org/"))) == 13000
    assert asyncio.run(b.fetch("https://x.org/")) == ""


def test_adapter_budget_prevents_requests(monkeypatch):
    b = backend(monkeypatch, [])
    for queries, k in [(["q"] * 4, 3), (["q"], 11), ([""], 3)]:
        with pytest.raises(ValueError):
            asyncio.run(b.search(queries, k))
    with pytest.raises(ValueError):
        asyncio.run(b.fetch("file:///secret"))
    assert not b._opener.calls


def test_missing_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(ValueError):
        TavilyBackend()


def test_async_model_does_not_block_loop():
    started, release = threading.Event(), threading.Event()

    class Backend:
        def generate_chat(self, messages, max_new_tokens):
            started.set()
            assert release.wait(2)
            assert messages == [{"role": "user", "content": "q"}]
            assert max_new_tokens == 512
            return "ok"

    async def scenario():
        task = asyncio.create_task(AsyncChat(Backend())("q"))
        try:
            for _ in range(200):
                if started.is_set():
                    break
                await asyncio.sleep(0.001)
            assert started.is_set() and not task.done()
        finally:
            release.set()
        assert await task == "ok"

    asyncio.run(scenario())


def test_tavily_transport_runs_off_event_loop(monkeypatch):
    b = backend(monkeypatch, [])
    loop_thread = threading.get_ident()

    def post(endpoint, body):
        assert threading.get_ident() != loop_thread
        return {"results": []}

    b._post = post
    assert asyncio.run(b.search(["q"], 3)) == []


def test_transport_byte_budget(monkeypatch):
    b = backend(monkeypatch, [])

    class Large:
        def open(self, request, timeout):
            return Response(b"x" * 2_000_001)

    b._opener = Large()
    with pytest.raises(RuntimeError):
        asyncio.run(b.search(["q"], 3))


def test_no_redirect_handler():
    from lychee_mas.backends.tavily_backend import _NoRedirect

    assert _NoRedirect().redirect_request(None, None, 302, "", {}, "https://x") is None


def test_smoke_reuses_backend_with_no_hidden_retries(monkeypatch):
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace

    script = Path(__file__).parents[1] / "scripts" / "smoke_mindsearch_searcher.py"
    spec = importlib.util.spec_from_file_location("searcher_smoke_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    class Client:
        def with_options(self, **kwargs):
            assert kwargs["max_retries"] == 0
            assert kwargs["http_client"].follow_redirects is False
            kwargs["http_client"].close()
            return self

        def close(self):
            pass

    class Backend:
        def __init__(self, **kwargs):
            assert kwargs["api_key_env"] == "SILICONFLOW_API_KEY"
            assert kwargs["timeout"] == 30.0
            self.client = Client()

        def generate_chat(self, messages, max_new_tokens):
            assert max_new_tokens == 512
            calls.append(messages)
            outputs = [{"queries": ["q", "q2"]}, {"source_ids": ["S1"]},
                       {"answer": "Warsaw [S1]", "source_ids": ["S1"],
                        "evidence": [{"source_id": "S1", "quote": "Warsaw"}]}]
            return SimpleNamespace(text=json.dumps(outputs[len(calls)-1]),
                                   n_prompt_pos=2, n_gen_tokens=3)

    monkeypatch.setattr(module, "OpenAICompatibleBackend", Backend)
    b = backend(monkeypatch, [
        {"results": [{"url": "https://x.org/", "title": "Curie", "content": "Warsaw"}]},
        {"results": [{"url": "https://x.org/", "raw_content": "Warsaw"}]}])
    monkeypatch.setattr(module, "TavilyBackend", lambda: b)
    assert asyncio.run(module.run("q")) == 0
    assert len(calls) == 3 and len(b._opener.calls) == 2


def test_tavily_trace_before_and_after_truncation(monkeypatch, caplog):
    caplog.set_level("INFO", logger="lychee_mas.mindsearch.trace")
    b = backend(monkeypatch, [{"results": [
        {"url": "https://x.org/", "raw_content": "x" * 13000}]}])
    assert len(asyncio.run(b.fetch("https://x.org/"))) == 13000
    event = next(json.loads(r.getMessage()) for r in caplog.records
                 if json.loads(r.getMessage())["event"] == "tavily.extract")
    assert event["raw_chars"] == 13000 and event["returned_chars"] == 13000
    assert event["truncated"] is False


def test_api_backend_preserves_finish_reason_without_network(monkeypatch):
    import sys
    from types import SimpleNamespace

    from lychee_mas.backends.openai_api_backend import OpenAICompatibleBackend

    def create(**kwargs):
        assert kwargs["max_tokens"] == 512
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"),
                                     finish_reason="length")],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=512))

    def client(**kwargs):
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=client))
    model = OpenAICompatibleBackend("fake-model", api_key="fake-key")
    result = model.generate_chat([{"role": "user", "content": "q"}], max_new_tokens=512)
    assert result.finish_reason == "length" and result.text == "{}"
    assert result.n_prompt_pos == 12 and result.n_gen_tokens == 512


def test_extract_raw_bound_does_not_grow_model_budget(monkeypatch):
    b = backend(monkeypatch, [{"results": [
        {"url": "https://x.org/", "raw_content": "x" * 300001}]}])
    assert len(asyncio.run(b.fetch("https://x.org/"))) == 300000
    with pytest.raises(ValueError):
        TavilyBackend(max_page_chars=300001)
