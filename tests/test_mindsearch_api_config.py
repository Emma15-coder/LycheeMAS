"""只用临时.env与假SDK，不读取项目真实Key、不发送请求。"""
import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("dotenv")
CONFIG_ENV_NAMES = (
    "SILICONFLOW_API_KEY", "TAVILY_API_KEY", "LYCHEE_LLM_API_KEY",
    "LYCHEE_LLM_API_KEY_ENV", "LYCHEE_LLM_MODEL", "LYCHEE_LLM_BASE_URL",
)


@pytest.fixture
def entry(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for name in CONFIG_ENV_NAMES + ("CUSTOM_API_KEY",):
        monkeypatch.delenv(name, raising=False)
    path = Path(__file__).parents[1] / "scripts/run_mindsearch.py"
    spec = importlib.util.spec_from_file_location("mindsearch_api_config_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ENV_FILE", tmp_path / ".env")
    return module


def args(**kw):
    return SimpleNamespace(**{**dict(model=None, base_url=None, api_key_env=None,
                                     arm="mindsearch", searcher="real", engine="tavily"), **kw})


def write_env(entry, text="SILICONFLOW_API_KEY=fake-model\nTAVILY_API_KEY=fake-search\n"):
    entry.ENV_FILE.write_text(text)
    entry.ENV_FILE.chmod(0o600)


def test_project_env_and_verified_defaults(entry):
    write_env(entry)
    a = args()
    entry.configure_api(a)
    assert a.model == "deepseek-ai/DeepSeek-V4.1-Flash"
    assert a.base_url == "https://api.siliconflow.cn/v1"
    assert a.api_key_env == "SILICONFLOW_API_KEY" and a.env_file_loaded


def test_cli_overrides_process_overrides_file(entry, monkeypatch):
    write_env(entry, "SILICONFLOW_API_KEY=file-key\nTAVILY_API_KEY=file-search\n"
              "LYCHEE_LLM_MODEL=file-model\nLYCHEE_LLM_BASE_URL=https://file.invalid/v1\n")
    monkeypatch.setenv("SILICONFLOW_API_KEY", "process-key")
    monkeypatch.setenv("LYCHEE_LLM_MODEL", "process-model")
    monkeypatch.setenv("LYCHEE_LLM_BASE_URL", "https://process.invalid/v1")
    a = args()
    entry.configure_api(a)
    assert a.model == "process-model" and a.base_url == "https://process.invalid/v1"
    assert entry.os.environ["SILICONFLOW_API_KEY"] == "process-key"
    monkeypatch.setenv("CUSTOM_API_KEY", "custom-key")
    b = args(model="cli-model", base_url="https://cli.invalid/v1", api_key_env="CUSTOM_API_KEY")
    entry.configure_api(b)
    assert (b.model, b.base_url, b.api_key_env) == (
        "cli-model", "https://cli.invalid/v1", "CUSTOM_API_KEY")


def test_process_only_and_legacy_generic_key(entry, monkeypatch):
    monkeypatch.setenv("LYCHEE_LLM_API_KEY", "legacy-key")
    a = args(arm="no_search")
    entry.configure_api(a)
    assert a.api_key_env == "LYCHEE_LLM_API_KEY" and not a.env_file_loaded


def test_custom_key_in_file(entry):
    write_env(entry, "CUSTOM_API_KEY=custom\nTAVILY_API_KEY=search\n")
    a = args(api_key_env="CUSTOM_API_KEY")
    entry.configure_api(a)
    assert entry.os.environ[a.api_key_env] == "custom"


@pytest.mark.parametrize("problem", ["permission", "format", "encoding", "symlink"])
def test_bad_file_fails_without_disclosing_content(entry, problem):
    write_env(entry)
    if problem == "permission":
        entry.ENV_FILE.chmod(0o644)
    elif problem == "format":
        entry.ENV_FILE.write_text("not-valid-private-content")
    elif problem == "encoding":
        entry.ENV_FILE.write_bytes(b"private-content-\xff")
    else:
        target = entry.ENV_FILE.with_name("actual")
        entry.ENV_FILE.rename(target)
        entry.ENV_FILE.symlink_to(target)
    with pytest.raises(SystemExit) as exc:
        entry.configure_api(args())
    assert "private-content" not in str(exc.value) and "fake-model" not in str(exc.value)


def test_missing_tavily_fails_before_creating_model(entry):
    write_env(entry, "SILICONFLOW_API_KEY=fake\n")
    with pytest.raises(SystemExit, match="TAVILY_API_KEY"):
        entry.configure_api(args())
    entry.configure_api(args(arm="no_search"))


def test_check_config_never_constructs_backend_or_prints_keys(entry, monkeypatch, capsys):
    write_env(entry)
    monkeypatch.setattr(sys, "argv", ["run_mindsearch.py", "--searcher", "real",
                                           "--check-config"])
    monkeypatch.setattr(entry, "build_llm", lambda *a: pytest.fail("must not create model"))
    entry.main()
    output = capsys.readouterr().out
    assert "fake-model" not in output and "fake-search" not in output
    assert json.loads(output)["api_key_env"] == "SILICONFLOW_API_KEY"


def test_build_llm_forwards_configuration_and_disables_sdk_retries(entry, monkeypatch):
    from lychee_mas.backends import openai_api_backend

    captured = {}

    class Backend:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.client = self

        def with_options(self, **kwargs):
            assert kwargs == {"max_retries": 0}
            return self

        def generate_chat(self, messages, max_new_tokens):
            assert messages == [{"role": "user", "content": "q"}]
            assert max_new_tokens == 512
            return SimpleNamespace(text="ok")

    monkeypatch.setattr(openai_api_backend, "OpenAICompatibleBackend", Backend)
    a = args(model="test-model", base_url="https://test.invalid/v1", api_key_env="TEST_KEY")
    a.timeout, a.temperature, a.max_new_tokens = 30, 0, 512
    model = entry.build_llm(a)
    assert captured == dict(model="test-model", base_url="https://test.invalid/v1",
                            api_key_env="TEST_KEY", timeout=30, temperature=0)
    assert asyncio.run(model("q")).text == "ok"


@pytest.mark.parametrize("arm", ["no_search", "single_turn"])
def test_main_auto_loads_keys_and_writes_standard_artifacts_without_network(
    entry, monkeypatch, tmp_path, capsys, arm,
):
    write_env(entry)
    collected = []

    def build_model(a):
        collected.append((a.model, a.base_url, a.api_key_env))

        async def model(prompt):
            entry._LLM_CALLS["n"] += 1
            return SimpleNamespace(text="Answer: test-answer", n_prompt_pos=10, n_gen_tokens=2)

        return model

    monkeypatch.setattr(entry, "build_llm", build_model)
    monkeypatch.setattr(entry, "_git", lambda *args: "test-git")
    out = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["run_mindsearch.py", "--arm", arm,
                                           "--question", "test-question", "--out-root", str(out),
                                           "--searcher-trace"])
    entry.main()
    folder = next((out / "adhoc" / arm).iterdir())
    config = json.loads((folder / "run_config.json").read_text())
    assert config["api_key_env"] == "SILICONFLOW_API_KEY" and config["env_file_loaded"]
    assert config["searcher_trace"] is True
    assert (folder / "searcher_trace.jsonl").stat().st_mode & 0o777 == 0o600
    assert collected == [("deepseek-ai/DeepSeek-V4.1-Flash", "https://api.siliconflow.cn/v1",
                          "SILICONFLOW_API_KEY")]
    sample = json.loads((folder / "predictions.jsonl").read_text())
    assert sample["prediction"] == "test-answer"
    assert sample["method"] == sample["arm"] == arm
    assert sample["planner_prompt_tokens"] == sample["prompt_tokens"] == 10
    assert sample["planner_completion_tokens"] == sample["completion_tokens"] == 2
    assert sample["searcher_prompt_tokens"] == sample["searcher_completion_tokens"] == 0
    assert sample["queries_count"] == sample["pages_read"] == (arm == "single_turn")
    metrics = json.loads((folder / "metrics.json").read_text())
    assert metrics["accuracy"] is None  # 手输单题没有标准答案
    assert metrics["avg_prompt_tokens"] == 10
    assert metrics["avg_completion_tokens"] == 2
    assert metrics["total_planner_prompt_tokens"] == 10
    assert metrics["avg_queries_count"] == (arm == "single_turn")
    expected = entry.M.aggregate_samples([sample], run_info={
        "model": config["model"], "method": arm, "task": "adhoc",
        "scorer_kind": "exact", "git_sha": config["git_sha"],
    })
    assert metrics["framework"] == expected
    assert metrics["framework"]["method"] == arm
    assert metrics["framework"]["total_input_total_positions"] == 10
    assert metrics["framework"]["total_output_text_tokens"] == 2
    assert metrics["framework"]["total_model_calls"] == 1
    assert (folder / "metrics.json").exists() and (folder / "sources.jsonl").exists()
    assert "fake-model" not in (folder / "run_config.json").read_text()
    assert "fake-search" not in capsys.readouterr().out


def test_help_does_not_load_env(entry, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_mindsearch.py", "--help"])
    monkeypatch.setattr(entry, "configure_api", lambda *a: pytest.fail("help must not load keys"))
    with pytest.raises(SystemExit) as exc:
        entry.main()
    assert exc.value.code == 0



def test_dotenv_supports_export_quotes_comments_and_expansion(entry):
    write_env(entry, '# standard dotenv syntax\n'
              'export SILICONFLOW_API_KEY="fake-model"\n'
              "TAVILY_API_KEY='fake-search' # inline comment\n"
              'MODEL_ALIAS=test-model\n'
              'LYCHEE_LLM_MODEL="${MODEL_ALIAS}"\n')
    a = args()
    entry.configure_api(a)
    assert a.model == "test-model" and a.env_file_loaded
    assert entry.os.environ["SILICONFLOW_API_KEY"] == "fake-model"
    assert entry.os.environ["TAVILY_API_KEY"] == "fake-search"


def test_dotenv_expansion_respects_process_environment(entry, monkeypatch):
    monkeypatch.setenv("MODEL_ALIAS", "process-alias")
    write_env(entry, 'SILICONFLOW_API_KEY=fake\nTAVILY_API_KEY=fake\n'
              'MODEL_ALIAS=file-alias\nLYCHEE_LLM_MODEL=${MODEL_ALIAS}\n')
    a = args()
    entry.configure_api(a)
    assert a.model == "process-alias"


def test_load_dotenv_called_with_checked_stream_and_override_false(entry, monkeypatch):
    import dotenv

    write_env(entry)
    actual = dotenv.load_dotenv
    calls = []

    def tracked(*args, **kwargs):
        assert args == () and kwargs["override"] is False
        assert kwargs["stream"] is not None
        calls.append(True)
        return actual(*args, **kwargs)

    monkeypatch.setattr(dotenv, "load_dotenv", tracked)
    entry.configure_api(args())
    assert calls == [True]


@pytest.mark.parametrize('status', ['ok', 'error'])
def test_single_turn_splits_answer_and_search_costs_even_after_failed_search(entry, status):
    from lychee_mas.methods.build.mindsearch import SearcherResult

    class Searcher:
        async def run(self, question, context):
            return SearcherResult(sub_question=question, status=status, answer='',
                                  queries=['q1', 'q2'], pages_read=1,
                                  prompt_tokens=17, completion_tokens=5)

    async def model(prompt):
        return SimpleNamespace(text='Answer: unknown', n_prompt_pos=11, n_gen_tokens=3)

    result = asyncio.run(entry.run_single_turn(Searcher(), model, 'question'))
    assert result['planner_prompt_tokens'] == 11
    assert result['planner_completion_tokens'] == 3
    assert result['searcher_prompt_tokens'] == 17
    assert result['searcher_completion_tokens'] == 5
    assert result['prompt_tokens'] == 28 and result['completion_tokens'] == 8
    assert result['queries_count'] == 2 and result['pages_read'] == 1
