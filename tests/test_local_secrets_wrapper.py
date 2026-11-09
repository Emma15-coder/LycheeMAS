"""只使用临时目录和假Key，绝不读取用户真实密钥或启动真实服务。"""
import importlib.util
import os
import stat
from pathlib import Path

import pytest


@pytest.fixture
def wrapper():
    script = Path(__file__).parents[1] / "scripts" / "run_searcher_smoke.py"
    spec = importlib.util.spec_from_file_location("local_secrets_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(wrapper, tmp_path):
    path = tmp_path / "lychee-mas" / "secrets.env"
    path.parent.mkdir(mode=0o700)
    path.write_text("SILICONFLOW_API_KEY=fake-model\nTAVILY_API_KEY=fake-search\n")
    path.chmod(0o600)
    return path


def test_setup_outside_repo_and_permissions(wrapper, tmp_path, monkeypatch, capsys):
    keys = iter(("fake-model", "fake-search"))
    monkeypatch.setattr(wrapper.getpass, "getpass", lambda prompt: next(keys))
    path = tmp_path / "config" / "lychee-mas" / "secrets.env"
    wrapper._setup(path)
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert wrapper._load(path) == {"SILICONFLOW_API_KEY": "fake-model",
                                    "TAVILY_API_KEY": "fake-search"}
    output = capsys.readouterr()
    assert "fake-model" not in output.out and "fake-search" not in output.err


@pytest.mark.parametrize("target,mode", [("dir", 0o755), ("file", 0o644)])
def test_reject_open_permissions(wrapper, tmp_path, target, mode):
    path = prepare(wrapper, tmp_path)
    (path.parent if target == "dir" else path).chmod(mode)
    with pytest.raises(ValueError):
        wrapper._load(path)


def test_no_shell_expansion_or_execution(wrapper, tmp_path):
    path = prepare(wrapper, tmp_path)
    literal = "$(touch${IFS}/tmp/should-never-be-created)"
    path.write_text(f"SILICONFLOW_API_KEY={literal}\nTAVILY_API_KEY=fake\n")
    assert wrapper._load(path)["SILICONFLOW_API_KEY"] == literal


@pytest.mark.parametrize("text", [
    "SILICONFLOW_API_KEY=fake\n", "SILICONFLOW_API_KEY=\nTAVILY_API_KEY=fake\n",
    "SILICONFLOW_API_KEY=fake\nTAVILY_API_KEY=fake\nOTHER=bad\n",
    "SILICONFLOW_API_KEY=fake\nSILICONFLOW_API_KEY=again\nTAVILY_API_KEY=fake\n",
])
def test_invalid_format(wrapper, tmp_path, text):
    path = prepare(wrapper, tmp_path)
    path.write_text(text)
    with pytest.raises(ValueError):
        wrapper._load(path)


def test_reject_symlink(wrapper, tmp_path):
    path = prepare(wrapper, tmp_path)
    link = path.parent / "link.env"
    link.symlink_to(path)
    with pytest.raises(OSError):
        wrapper._load(link)


def test_smoke_fixed_question_and_environment_only_keys(wrapper, monkeypatch):
    captured = {}

    def fake_exec(python, argv, env):
        captured.update(python=python, argv=argv, env=env)

    monkeypatch.setattr(wrapper.os, "execve", fake_exec)
    monkeypatch.setenv("LYCHEE_LLM_BASE_URL", "https://unexpected.invalid")
    wrapper._smoke({"SILICONFLOW_API_KEY": "fake-model", "TAVILY_API_KEY": "fake-search"})
    assert captured["env"]["SILICONFLOW_API_KEY"] == "fake-model"
    assert captured["env"]["LYCHEE_LLM_BASE_URL"] == "https://api.siliconflow.cn/v1"
    assert captured["argv"][-2:] == ["--question", wrapper.QUESTION]
    assert "fake-model" not in str(captured["argv"])
    assert "fake-search" not in str(captured["argv"])


def test_setup_failure_does_not_overwrite_existing(wrapper, tmp_path, monkeypatch):
    path = prepare(wrapper, tmp_path)
    old = path.read_bytes()
    monkeypatch.setattr(wrapper.getpass, "getpass", lambda prompt: "")
    with pytest.raises(ValueError):
        wrapper._setup(path)
    assert path.read_bytes() == old


def test_check_mode_no_network_or_exec(wrapper, tmp_path, monkeypatch, capsys):
    path = prepare(wrapper, tmp_path)
    monkeypatch.setattr(wrapper, "_location", lambda: path)
    monkeypatch.setattr(wrapper.sys, "argv", ["wrapper", "check"])
    monkeypatch.setattr(os, "execve", lambda *args: pytest.fail("must not execute"))
    assert wrapper.main() == 0
    output = capsys.readouterr()
    assert "fake-model" not in output.out and "fake-search" not in output.out


def test_invalid_encoding_error_does_not_disclose_file_bytes(
    wrapper, tmp_path, monkeypatch, capsys,
):
    path = prepare(wrapper, tmp_path)
    path.write_bytes(b"SILICONFLOW_API_KEY=private-\xff\n")
    monkeypatch.setattr(wrapper, "_location", lambda: path)
    monkeypatch.setattr(wrapper.sys, "argv", ["wrapper", "check"])
    assert wrapper.main() == 1
    output = capsys.readouterr()
    assert "private" not in output.err and "xff" not in output.err


def test_wrapper_does_not_accept_arbitrary_commands(wrapper, monkeypatch):
    monkeypatch.setattr(wrapper.sys, "argv", ["wrapper", "eval", "--count", "100"])
    with pytest.raises(SystemExit) as exc:
        wrapper.main()
    assert exc.value.code == 2
