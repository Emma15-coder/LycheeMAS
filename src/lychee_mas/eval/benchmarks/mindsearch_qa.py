"""MindSearch closed-set QA loaders: Bamboogle / MuSiQue / HotpotQA.

论文 §3.2（zero-shot）：Bamboogle (Press et al., 2022)、MuSiQue (Trivedi et al., 2022)、
HotpotQA (Yang et al., 2018)。论文未指明 HotpotQA 子设置；本实现取 ``fullwiki``——
MindSearch 是开放域检索，不消费官方提供的段落，distractor / fullwiki 的题目与
难度分布一致，这里只使用 question / answer / level 三个字段。

三个数据集共用一套 prepare：下载并转成 parquet 落到
``prepared_root()/mindsearch/<dataset>/``，之后 ``load_*`` 只读本地。
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

from .common import load_parquet, parquet_ready, prepared_root, save_hf_dataset
from .source_catalog import provider_ids, source_backend_order

# dataset -> split / subset / 必需列；subset=None 表示 HF 无 config 维度
_SPEC: Dict[str, Dict] = {
    # q/a 是原始列名（Bamboogle 用大写 Question/Answer），columns 供缓存校验
    "bamboogle": {"split": "test", "subset": None,
                  "columns": ("Question", "Answer"), "q": "Question", "a": "Answer"},
    "musique": {"split": "validation", "subset": None,
                "columns": ("question", "answer"),
                "q": "question", "a": "answer"},
    "hotpotqa": {"split": "validation", "subset": "fullwiki",
                 "columns": ("question", "answer", "level"),
                 "q": "question", "a": "answer"},
}


def _out_dir(dataset: str) -> str:
    return os.path.join(prepared_root(), "mindsearch", dataset)


def _ready(dataset: str) -> bool:
    spec = _SPEC[dataset]
    return parquet_ready(f"mindsearch/{dataset}", spec["split"],
                         required_columns=spec["columns"])


def prepare_mindsearch(dataset: str, force: bool = False, source: Optional[str] = None) -> str:
    """把 dataset 落到本地（已存在直接返回）；返回 prepared 目录。"""
    if dataset not in _SPEC:
        raise KeyError(f"unknown mindsearch dataset {dataset!r}; available: {sorted(_SPEC)}")
    out_dir = _out_dir(dataset)
    if not force and _ready(dataset):
        return out_dir
    errors: List[str] = []
    for backend in source_backend_order(dataset, source):
        for repo_id in provider_ids(dataset, backend):
            try:
                save_hf_dataset(repo_id, _SPEC[dataset]["subset"],
                                _SPEC[dataset]["split"], out_dir, benchmark=dataset)
                return out_dir
            except Exception as exc:  # pragma: no cover - network/provider dependent
                errors.append(f"{backend}:{repo_id}: {exc}")
    raise RuntimeError(f"failed to prepare {dataset}; " + (" | ".join(errors) or "no source"))


def _rows(dataset: str) -> List[Dict]:
    prepare_mindsearch(dataset)
    rows = list(load_parquet(f"mindsearch/{dataset}", _SPEC[dataset]["split"]))
    if not rows:
        raise ValueError(f"{dataset}: prepared split {_SPEC[dataset]['split']!r} is empty")
    return rows


def _record(task: str, question: str, gold: str) -> Dict:
    return {"task": task, "kind": "exact", "question": str(question).strip(),
            "gold": str(gold).strip(), "context": None}


def _hops(it: Dict) -> Optional[int]:
    """题目 hop 数：优先 len(question_decomposition)，兼容 JSON 字符串与 id 前缀。"""
    dec = it.get("question_decomposition")
    if isinstance(dec, str):
        try:
            dec = json.loads(dec)
        except ValueError:
            dec = None
    if isinstance(dec, (list, tuple)):
        return len(dec)
    match = re.match(r"(\d)hop", str(it.get("id", "")))
    return int(match.group(1)) if match else None


def _take(rows: List[Dict], n: Optional[int]) -> List[Dict]:
    return rows if n is None else rows[:n]


def load_bamboogle(n: Optional[int] = None) -> List[Dict]:
    q, a = _SPEC["bamboogle"]["q"], _SPEC["bamboogle"]["a"]
    rows = [_record("bamboogle", it[q], it[a]) for it in _rows("bamboogle")]
    return _take(rows, n)


def load_musique(n: Optional[int] = None, *, hops: int = 2) -> List[Dict]:
    task = f"musique_{hops}hop"
    q, a = _SPEC["musique"]["q"], _SPEC["musique"]["a"]
    rows = [_record(task, it[q], it[a]) for it in _rows("musique")
            if _hops(it) == hops]
    if not rows:
        raise ValueError(f"musique: no question with {hops} hops; check repo id / split")
    return _take(rows, n)


def load_hotpotqa(n: Optional[int] = None, *, level: str = "easy") -> List[Dict]:
    task = f"hotpotqa_{level}"
    q, a = _SPEC["hotpotqa"]["q"], _SPEC["hotpotqa"]["a"]
    rows = [_record(task, it[q], it[a]) for it in _rows("hotpotqa")
            if it.get("level") == level]
    if not rows:
        raise ValueError(f"hotpotqa: no question with level={level!r}; check repo id / split")
    return _take(rows, n)
