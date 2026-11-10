"""MindSearch 数据集 loader 的离线测试：不联网、不依赖 datasets。"""

from __future__ import annotations

import pytest

from lychee_mas.eval import benchmarks as B
from lychee_mas.eval.benchmarks import mindsearch_qa as MQ


def _fake_rows(monkeypatch, dataset: str, rows):
    monkeypatch.setattr(MQ, "_rows", lambda name: rows if name == dataset else [])


# Bamboogle 原始列名是大写 Question / Answer
BAMBOOGLE = [
    {"Question": " Q1 ", "Answer": " A1 "},
    {"Question": "Q2", "Answer": "A2"},
    {"Question": "Q3", "Answer": "A3"},
]

MUSIQUE = [
    {"question": "M2", "answer": "a2", "question_decomposition": [1, 2]},
    {"question": "M3", "answer": "a3", "question_decomposition": [1, 2, 3]},
    {"question": "M2b", "answer": "a2b", "question_decomposition": [1, 2]},
]

HOTPOT = [
    {"question": "H-easy", "answer": "e", "level": "easy"},
    {"question": "H-hard", "answer": "h", "level": "hard"},
    {"question": "H-medium", "answer": "m", "level": "medium"},
]


def test_bamboogle_mapping_and_slicing(monkeypatch):
    _fake_rows(monkeypatch, "bamboogle", BAMBOOGLE)
    rows = MQ.load_bamboogle(2)
    assert [r["question"] for r in rows] == ["Q1", "Q2"]      # 去空白 + 固定切片
    assert [r["gold"] for r in rows] == ["A1", "A2"]
    assert all(r["task"] == "bamboogle" and r["kind"] == "exact" for r in rows)
    assert all(r["context"] is None for r in rows)


def test_musique_filters_by_hops(monkeypatch):
    _fake_rows(monkeypatch, "musique", MUSIQUE)
    two = MQ.load_musique(hops=2)
    assert [r["question"] for r in two] == ["M2", "M2b"]
    assert all(r["task"] == "musique_2hop" for r in two)
    # 过滤后为空 -> 显式报错（不静默返回空表）
    with pytest.raises(ValueError, match="no question with 4 hops"):
        MQ.load_musique(hops=4)


def test_hotpotqa_filters_by_level(monkeypatch):
    _fake_rows(monkeypatch, "hotpotqa", HOTPOT)
    hard = MQ.load_hotpotqa(level="hard")
    assert [r["question"] for r in hard] == ["H-hard"]
    assert hard[0]["task"] == "hotpotqa_hard"


def test_registry_routes_aliases(monkeypatch):
    _fake_rows(monkeypatch, "musique", MUSIQUE)
    rows = B.load("musique_3hop", 1)
    assert len(rows) == 1 and rows[0]["task"] == "musique_3hop"

    _fake_rows(monkeypatch, "hotpotqa", HOTPOT)
    assert B.load("hotpotqa_medium", 1)[0]["question"] == "H-medium"


def test_unknown_dataset_raises():
    try:
        MQ.prepare_mindsearch("nope")
    except KeyError as exc:
        assert "unknown mindsearch dataset" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("should raise")


def test_musique_hop_count_fallbacks(monkeypatch):
    """hop 数兼容三种来源：list、JSON 字符串、id 前缀。"""
    rows = [
        {"question": "S", "answer": "a",
         "question_decomposition": '[{"q": 1}, {"q": 2}]'},
        {"question": "I", "answer": "b", "id": "3hop__xyz"},
    ]
    _fake_rows(monkeypatch, "musique", rows)
    assert [r["question"] for r in MQ.load_musique(hops=2)] == ["S"]
    assert [r["question"] for r in MQ.load_musique(hops=3)] == ["I"]
