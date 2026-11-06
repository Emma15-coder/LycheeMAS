"""诊断捕获和重放只用假回调，不读取本地密钥。"""
import asyncio
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def diagnostic(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("diagnostic_test", scripts /
                                                  "diagnose_mindsearch_searcher.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capture_redacts_full_material(diagnostic, tmp_path, monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "private-model-test")
    path = tmp_path / "capture.json"
    capture = diagnostic.Capture(path, "question")
    capture.data["models"] = [{"prompt": "a private-model-test prompt",
                                "response": {"text": "private-model-test"}}]
    capture.save()
    text = path.read_text()
    assert "private-model-test" not in text and "[REDACTED]" in text
    assert path.stat().st_mode & 0o777 == 0o600


def test_captured_full_run_can_replay_without_keys(diagnostic, tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "_load", lambda path: {})

    @dataclass
    class Response:
        text: str
        n_prompt_pos: int = 3
        n_gen_tokens: int = 2
        finish_reason: str = "stop"

    class Model:
        backend = SimpleNamespace(client=SimpleNamespace(close=lambda: None))

        async def __call__(self, prompt):
            if prompt.startswith("Rewrite"):
                return Response('{"queries":["test query"]}')
            if prompt.startswith("Select"):
                return Response('{"source_ids":["S1"]}')
            return Response('{"answer":"","source_ids":[],"evidence":[]}')

    async def search(*args):
        return [{"url": "https://x.org/", "title": "title", "snippet": "snippet"}]

    async def fetch(url):
        return "No evidence for the question."

    from lychee_mas.methods.build.mindsearch.searcher import WebSearcher

    monkeypatch.setattr(diagnostic, "make_searcher", lambda: WebSearcher(
        Model(), search, fetch, max_queries=1, max_read=1))
    path = tmp_path / "run.json"
    assert asyncio.run(diagnostic.execute(path, "question")) == 1
    capture = json.loads(path.read_text())
    assert capture["counts"] == {"model": 3, "search": 1, "fetch": 1}
    assert all("prompt" in i and "response" in i for i in capture["models"])
    assert capture["fetch"][0]["text"] == "No evidence for the question."
    assert asyncio.run(diagnostic.replay(path)) == 0
