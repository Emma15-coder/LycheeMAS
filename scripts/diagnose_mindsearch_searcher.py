"""有预算的单题诊断与完整脱敏捕获；run会调用服务，replay只复用本地材料。"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from lychee_mas.backends.searcher_trace import redact_fields
from lychee_mas.methods.build.mindsearch.searcher import WebSearcher
from run_searcher_smoke import QUESTION, _load, _location
from smoke_mindsearch_searcher import make_searcher

OTHER_QUESTION = "In which city is the Eiffel Tower located?"


class Capture:
    def __init__(self, path: Path, question: str) -> None:
        self.path = path
        self.data = {"question": question, "models": [], "search": [], "fetch": []}
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    def save(self) -> None:
        raw = json.dumps(redact_fields(self.data), ensure_ascii=False, indent=2)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(raw)


async def execute(path: Path, question: str, reuse: Path | None = None) -> int:
    os.environ.update(_load(_location()))
    os.environ["LYCHEE_LLM_BASE_URL"] = "https://api.siliconflow.cn/v1"
    os.environ["LYCHEE_LLM_MODEL"] = "deepseek-ai/DeepSeek-V4.1-Flash"
    capture = Capture(path, question)
    cached = json.loads(reuse.read_text()) if reuse else None
    searcher = make_searcher()
    llm, search, fetch = searcher.llm, searcher.search, searcher.fetch
    start = time.perf_counter()
    counts = {"model": 0, "search": 0, "fetch": 0}

    async def model(prompt):
        if counts["model"] >= 3:
            raise RuntimeError("model request budget exhausted")
        counts["model"] += 1
        item = {"prompt": prompt}
        capture.data["models"].append(item)
        capture.save()
        result = await llm(prompt)
        item["response"] = asdict(result)
        capture.save()
        return result

    async def searching(queries, topk):
        if cached:
            entry = next((i for i in cached["search"] if i["queries"] == queries), None)
            if entry is None:
                raise RuntimeError("cache miss; no unplanned search")
            rows = entry["rows"]
        else:
            if counts["search"] >= 1 or len(queries) != 1:
                raise RuntimeError("search request budget exhausted")
            counts["search"] += 1
            rows = await search(queries, topk)
        capture.data["search"].append({"queries": queries, "rows": rows})
        capture.save()
        return rows

    async def fetching(url):
        if cached:
            entry = next((i for i in cached["fetch"] if i["url"] == url), None)
            if entry is None:
                raise RuntimeError("cache miss; no unplanned extract")
            text = entry["text"]
        else:
            if counts["fetch"] >= 1:
                raise RuntimeError("fetch request budget exhausted")
            counts["fetch"] += 1
            text = await fetch(url)
        capture.data["fetch"].append({"url": url, "text": text})
        capture.save()
        return text

    searcher.llm, searcher.search, searcher.fetch = model, searching, fetching
    try:
        result = await searcher.run(question)
        capture.data.update(result=asdict(result), counts=counts,
                            elapsed_s=round(time.perf_counter() - start, 3))
        capture.save()
        print(json.dumps(redact_fields({"artifact": str(path), "result": asdict(result),
                                        "counts": counts,
                                        "elapsed_s": capture.data["elapsed_s"]}),
                         ensure_ascii=False, indent=2))
        return 0 if result.is_ok() else 1
    finally:
        llm.backend.client.close()


async def replay(path: Path) -> int:
    data = json.loads(path.read_text())
    calls = iter(data["models"])

    async def model(prompt):
        entry = next(calls)
        if prompt != entry["prompt"]:
            raise ValueError("prompt differs from captured run")
        return SimpleNamespace(**entry["response"])

    async def search(queries, topk):
        return next(i["rows"] for i in data["search"] if i["queries"] == queries)

    async def fetch(url):
        return next(i["text"] for i in data["fetch"] if i["url"] == url)

    result = await WebSearcher(model, search, fetch, topk=3, max_queries=1, max_read=1).run(
        data["question"])
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if asdict(result) == data["result"] else 1


def main() -> None:
    # 只影响此独立诊断进程，不改变离线测试调用execute/replay时的日志配置。
    import logging
    logging.disable(logging.CRITICAL)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "replay"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--case", choices=("original", "other"), default="original")
    parser.add_argument("--reuse-web", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "replay":
            code = asyncio.run(replay(args.path))
        else:
            if args.path.exists():
                raise ValueError("capture already exists; refusing duplicate run")
            code = asyncio.run(execute(args.path, QUESTION if args.case == "original"
                                       else OTHER_QUESTION, args.reuse_web))
    except Exception as exc:
        print(f"diagnosis failed ({type(exc).__name__}); details redacted")
        code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()
