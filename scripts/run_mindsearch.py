"""MindSearch 端到端实验入口（三臂：no_search / single_turn / mindsearch）。

用法：
  # 假检索器打通链路（模型仍真实调用，需要模型 Key；完全离线联调见 tests/）
  PYTHONPATH=src python scripts/run_mindsearch.py --arm mindsearch \
      --searcher stub --question "测试问题"

  # 真实模型 + 真检索（自动加载项目.env）
  PYTHONPATH=src python scripts/run_mindsearch.py --arm mindsearch \
      --searcher real --question "埃菲尔铁塔位于哪个城市？"

产物（runs/mindsearch/<task>/<arm>/<ts>/）：
  predictions.jsonl / dag/<case>.json / dag/<case>.mmd
  sources.jsonl / run_config.json / metrics.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from lychee_mas.backends.local_env import checked_env_stream
from lychee_mas.backends.searcher_trace import redact_fields, trace_file
from lychee_mas.eval import benchmarks as BENCH
from lychee_mas.eval import metrics as M
from lychee_mas.methods.build.mindsearch import MindSearchState, TaskDAG, planner_node_factory
from lychee_mas.methods.build.mindsearch.answer_format import format_answer
from lychee_mas.methods.build.mindsearch.searcher import SearcherResult, SourceRef
from lychee_mas.plugins import build_langgraph

NO_SEARCH_PROMPT = """请直接回答下面的问题，不要使用任何外部检索。
第一行必须是 `Answer: <短答案>`（实体 / yes / no / 数字），第二行起写解释。

问题：{question}
"""

SINGLE_TURN_PROMPT = """根据下面的检索结果回答问题。只依据检索结果作答；
检索结果不足时说明缺失什么，不要编造答案，也不要编造来源链接。
第一行必须是 `Answer: <短答案>`，第二行起写解释。

问题：{question}

检索结果（来源：{src}）：
{evidence}
"""


ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
COST_FIELDS = ("planner_prompt_tokens", "planner_completion_tokens",
               "searcher_prompt_tokens", "searcher_completion_tokens",
               "queries_count", "pages_read")


def configure_api(args: argparse.Namespace) -> None:
    """配置优先级：CLI > 进程环境 > 项目.env > 已验证服务默认值。"""
    try:
        with checked_env_stream(ENV_FILE, optional=True) as stream:
            if stream is None:
                env_loaded = False
            else:
                from dotenv import load_dotenv  # 配置功能按需导入，核心注册链不依赖它

                env_loaded = load_dotenv(stream=stream, override=False)
    except ImportError:
        raise SystemExit("需要python-dotenv；请安装项目的config或searcher可选依赖。") from None
    except (OSError, UnicodeError, ValueError):
        raise SystemExit(
            "项目.env读取失败；请检查格式、所有权和600权限（未显示敏感内容）。") from None
    args.model = args.model or os.environ.get("LYCHEE_LLM_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash")
    args.base_url = args.base_url or os.environ.get("LYCHEE_LLM_BASE_URL",
                                                   "https://api.siliconflow.cn/v1")
    args.api_key_env = (args.api_key_env or os.environ.get("LYCHEE_LLM_API_KEY_ENV")
                        or ("LYCHEE_LLM_API_KEY" if os.environ.get("LYCHEE_LLM_API_KEY")
                            else "SILICONFLOW_API_KEY"))
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.api_key_env):
        raise SystemExit("--api-key-env必须是环境变量名称，不能填写密钥值。")
    args.env_file_loaded = env_loaded
    if not os.environ.get(args.api_key_env):
        raise SystemExit("缺少指定模型密钥；请检查.env或--api-key-env对应的进程环境变量。")
    if (args.arm != "no_search" and args.searcher == "real" and args.engine == "tavily"
            and not os.environ.get("TAVILY_API_KEY")):
        raise SystemExit("缺少 TAVILY_API_KEY；请配置项目.env或进程环境。")


# LLM 调用计数（每题重置；用于框架指标 num_model_calls）
_LLM_CALLS = {"n": 0}


# ---------------------------------------------------------------- 依赖构造

def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return ""


def build_llm(args: argparse.Namespace):
    """OpenAI 兼容端点 → async callable（planner 与 searcher 共用同一个对象）。"""
    from lychee_mas.backends.openai_api_backend import OpenAICompatibleBackend

    backend = OpenAICompatibleBackend(
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        timeout=args.timeout,
        temperature=args.temperature,
    )

    # 关闭SDK自动重试；Planner的显式规划修正次数仍由规划参数控制。
    backend.client = backend.client.with_options(max_retries=0)

    async def llm(prompt: str):
        _LLM_CALLS["n"] += 1
        # generate_chat 是同步的：必须丢线程，否则并发检索会堵住事件循环
        return await asyncio.to_thread(
            backend.generate_chat, [{"role": "user", "content": prompt}], args.max_new_tokens)

    return llm


class StubSearcher:
    """离线替身：不联网，固定返回一条摘要，用来打通链路。"""

    def __init__(self, prefix: str = "[stub]") -> None:
        self.prefix = prefix

    async def run(self, sub_question: str, context: str = "") -> SearcherResult:
        await asyncio.sleep(0)
        return SearcherResult(
            sub_question=sub_question, status="ok",
            answer=f"{self.prefix} 关于「{sub_question}」的结论",
            sources=[SourceRef(url="https://example.com/stub", title="stub")],
            queries=[sub_question], pages_read=1)


def build_searcher(llm, args: argparse.Namespace):
    if args.searcher == "stub":
        return StubSearcher()
    # 真实检索：工厂由 B 提供，位于 methods 侧（与 WebSearcher 同文件）；
    # backends/ 只放搜索/抓取原语，脚本不直接依赖它（见 IFACE v1.7 §3.4）。
    from lychee_mas.methods.build.mindsearch import searcher as searcher_mod

    factory = getattr(searcher_mod, "make_web_searcher", None)
    if factory is None:
        raise SystemExit(
            "真实检索需要 methods/build/mindsearch/searcher.py 提供 "
            "make_web_searcher(llm=..., engine=..., topk=..., max_queries=..., max_read=...)"
            "（IFACE v1.7，由 B 实现，尚未合入）。\n"
            "先用 --searcher stub 打通链路。")
    searcher = factory(llm=llm, engine=args.engine, topk=args.topk,
                       max_queries=args.max_queries, max_read=args.max_read)
    # 与模型SDK的--timeout一致；整体Searcher仍受180秒上限约束。
    searcher.MODEL_TIMEOUT_S = min(getattr(args, "timeout", searcher.MODEL_TIMEOUT_S),
                                 searcher.RUN_TIMEOUT_S)
    return searcher


# ---------------------------------------------------------------- 三臂执行

async def run_mindsearch(app, question: str):
    state = await app.ainvoke({"question": question})
    answer = (state.get("answer") or "").strip()
    raw = (state.get("outputs") or {}).get("planner", "")
    if not answer:                                  # 兜底：从长答案里再抽一次
        answer = format_answer(raw).short
    return {
        "answer": answer,
        "answer_raw": raw,
        "dag": state.get("dag") or {},
        "turns": int(state.get("turns", 0) or 0),
        "truncated": bool(state.get("truncated")),
        "stop_reason": state.get("stop_reason", ""),
        "format_ok": bool(state.get("format_ok")),
        "citation_ok": bool(state.get("citation_ok")) if "citation_ok" in state else None,
        "citation_unknown": list(state.get("citation_unknown") or []),
        "prompt_tokens": int(state.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(state.get("completion_tokens", 0) or 0),
        **{key: int(state.get(key, 0) or 0) for key in COST_FIELDS},
    }


async def run_no_search(llm, question: str):
    res = await llm(NO_SEARCH_PROMPT.format(question=question))
    fmt = format_answer(res.text)
    return {"answer": fmt.short, "answer_raw": fmt.long, "dag": {}, "turns": 1,
            "truncated": False, "stop_reason": "no_search", "format_ok": fmt.format_ok,
            "citation_ok": None, "citation_unknown": [],
            "prompt_tokens": res.n_prompt_pos, "completion_tokens": res.n_gen_tokens,
            "planner_prompt_tokens": res.n_prompt_pos,
            "planner_completion_tokens": res.n_gen_tokens,
            "searcher_prompt_tokens": 0, "searcher_completion_tokens": 0,
            "queries_count": 0, "pages_read": 0}


async def run_single_turn(searcher, llm, question: str):
    """单轮基线：1 次检索 + 1 次作答（不建 DAG，也不做多轮规划）。"""
    res = await searcher.run(question, f"根问题：{question}")
    urls = []
    for src in res.sources or []:
        url = src.get("url") if isinstance(src, dict) else getattr(src, "url", "")
        if url:
            urls.append(str(url))
    prompt = SINGLE_TURN_PROMPT.format(question=question,
                                       src=", ".join(urls) or "（无）",
                                       evidence=res.answer or "（未检索到相关信息）")
    out = await llm(prompt)
    fmt = format_answer(out.text)
    return {"answer": fmt.short, "answer_raw": fmt.long, "dag": {}, "turns": 1,
            "truncated": False, "stop_reason": f"single_turn:{res.status}",
            "format_ok": fmt.format_ok, "citation_ok": None, "citation_unknown": [],
            "prompt_tokens": int(getattr(out, "n_prompt_pos", 0) or 0)
            + int(res.prompt_tokens or 0),
            "completion_tokens": int(getattr(out, "n_gen_tokens", 0) or 0)
            + int(res.completion_tokens or 0),
            "planner_prompt_tokens": int(getattr(out, "n_prompt_pos", 0) or 0),
            "planner_completion_tokens": int(getattr(out, "n_gen_tokens", 0) or 0),
            "searcher_prompt_tokens": int(res.prompt_tokens or 0),
            "searcher_completion_tokens": int(res.completion_tokens or 0),
            "queries_count": len(res.queries or []), "pages_read": int(res.pages_read or 0)}

async def run_ReAct(searcher, llm, question: str):
    pass

# ---------------------------------------------------------------- 产物

def dag_depth(dag: dict) -> int:
    depth: dict[str, int] = {}
    for node in dag.get("nodes", []):               # to_dict 已按拓扑序输出
        depth[node["qid"]] = 1 + max((depth.get(p, 0) for p in node["parents"]), default=0)
    return max(depth.values(), default=0)


def write_case_artifacts(out_dir, case_id: str, dag: dict) -> None:
    if not dag:
        return
    dag_dir = out_dir / "dag"
    dag_dir.mkdir(parents=True, exist_ok=True)
    (dag_dir / f"{case_id}.json").write_text(
        json.dumps(dag, ensure_ascii=False, indent=2), encoding="utf-8")
    (dag_dir / f"{case_id}.mmd").write_text(
        TaskDAG.from_dict(dag).to_mermaid(), encoding="utf-8")


def write_sources(out_dir, case_id: str, dag: dict) -> None:
    with (out_dir / "sources.jsonl").open("a", encoding="utf-8") as fh:
        for node in dag.get("nodes", []):
            for src in node.get("sources") or []:
                fh.write(json.dumps({"case_id": case_id, "qid": node["qid"], **src},
                                    ensure_ascii=False) + "\n")


def summarize(samples: list[dict]) -> dict:
    scored = [s for s in samples if s["score"] is not None]
    cit = [s for s in samples if s.get("citation_ok") is not None]
    n = len(samples) or 1
    reasons = Counter(s["stop_reason"] for s in samples)
    return {
        "n": len(samples),
        "n_scored": len(scored),
        "accuracy": round(sum(s["score"] for s in scored) / len(scored), 4) if scored else None,
        "citation_valid_rate": (round(sum(1 for s in cit if s["citation_ok"]) / len(cit), 3)
                                if cit else None),
        "avg_prompt_tokens": round(sum(s["prompt_tokens"] for s in samples) / n, 1),
        "avg_completion_tokens": round(sum(s["completion_tokens"] for s in samples) / n, 1),
        **{f"total_{key}": sum(s.get(key, 0) for s in samples) for key in COST_FIELDS},
        **{f"avg_{key}": round(sum(s.get(key, 0) for s in samples) / n, 1)
           for key in COST_FIELDS},
        "avg_latency_s": round(sum(s["latency_s"] for s in samples) / n, 2),
        "avg_turns": round(sum(s["turns"] for s in samples) / n, 2),
        "avg_nodes": round(sum(s["n_nodes"] for s in samples) / n, 2),
        "avg_dag_depth": round(sum(s["dag_depth"] for s in samples) / n, 2),
        "format_ok_rate": round(sum(1 for s in samples if s["format_ok"]) / n, 3),
        "stop_reason": dict(reasons),
    }


# ---------------------------------------------------------------- 主流程

def main() -> None:
    ap = argparse.ArgumentParser(description="MindSearch 端到端实验")
    ap.add_argument("--arm", default="mindsearch",
                    choices=("mindsearch", "no_search", "single_turn"))
    ap.add_argument("--searcher", default="stub", choices=("stub", "real"))
    ap.add_argument("--engine", default="tavily",
                    help="真实检索后端：tavily | ddg | wiki（由 B 的 make_web_searcher 实现）")
    ap.add_argument("--benchmark", default=None, help="benchmark 注册名（如 bamboogle）")
    ap.add_argument("--question", default=None, help="只跑单题（调试用）")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-env", default=None)
    ap.add_argument("--check-config", action="store_true", help="仅验证配置，不调用服务")
    ap.add_argument("--searcher-trace", action="store_true",
                    help="保存脱敏Searcher阶段日志、提示词、模型返回和正文，供离线诊断")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--max-nodes", type=int, default=40)
    ap.add_argument("--max-subq", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-fail-streak", type=int, default=2)
    ap.add_argument("--min-nodes-before-finish", type=int, default=1)
    ap.add_argument("--max-premature-retries", type=int, default=2)
    ap.add_argument("--max-citation-retries", type=int, default=1)
    ap.add_argument("--max-redundant", type=int, default=2)
    ap.add_argument("--require-citation", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--max-queries", type=int, default=3)
    ap.add_argument("--max-read", type=int, default=3)
    ap.add_argument("--out-root", default="runs/mindsearch")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    configure_api(args)
    if args.check_config:
        print(json.dumps(redact_fields({
            "model": args.model, "base_url": args.base_url, "api_key_env": args.api_key_env,
            "env_file_loaded": args.env_file_loaded, "model_key_present": True,
            "tavily_key_present": bool(os.environ.get("TAVILY_API_KEY")),
            "arm": args.arm, "searcher": args.searcher,
        }), ensure_ascii=False, indent=2))
        return

    # ---- 数据集 ----
    if args.question:
        records = [{"task": "adhoc", "kind": "exact", "question": args.question,
                    "gold": "", "context": None}]
        args.benchmark = args.benchmark or "adhoc"
    else:
        if not args.benchmark:
            raise SystemExit("需要 --benchmark 或 --question")
        try:
            records = BENCH.load(args.benchmark, args.n)
        except KeyError as exc:
            raise SystemExit(
                f"数据集 {args.benchmark!r} 未注册（需要 B 的 eval 分支）：{exc}") from exc

    # ---- 组件 ----
    llm = build_llm(args)
    app = None
    searcher = None
    if args.arm == "mindsearch":
        searcher = build_searcher(llm, args)
        factory = planner_node_factory(
            searcher=searcher, llm=llm,
            max_turns=args.max_turns, max_nodes=args.max_nodes,
            max_subq_per_turn=args.max_subq, max_concurrency=args.concurrency,
            max_fail_streak=args.max_fail_streak,
            min_nodes_before_finish=args.min_nodes_before_finish,
            max_premature_retries=args.max_premature_retries,
            max_citation_retries=args.max_citation_retries,
            max_redundant=args.max_redundant,
            require_citation=args.require_citation)
        sg = build_langgraph(method="mindsearch", node_factory=factory,
                             state_schema=MindSearchState)
        app = sg.compile()
    elif args.arm == "single_turn":
        # 近似论文的 single-turn 基线：1 个 query、1 个结果、1 页
        single = argparse.Namespace(**vars(args))
        single.topk, single.max_queries, single.max_read = 1, 1, 1
        searcher = build_searcher(llm, single)

    # ---- 输出目录 ----
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
    tag = (args.benchmark or "adhoc").replace("/", "_")
    out_dir = __import__("pathlib").Path(args.out_root) / tag / args.arm / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sources.jsonl").touch()

    config = {
        "script": "run_mindsearch.py", "arm": args.arm, "searcher": args.searcher,
        "engine": args.engine,
        "benchmark": args.benchmark, "n": len(records), "seed": args.seed,
        "model": args.model, "base_url": args.base_url, "temperature": args.temperature,
        "api_key_env": args.api_key_env, "env_file_loaded": args.env_file_loaded,
        "searcher_trace": args.searcher_trace,
        "max_new_tokens": args.max_new_tokens, "max_turns": args.max_turns,
        "max_nodes": args.max_nodes, "max_subq_per_turn": args.max_subq,
        "max_concurrency": args.concurrency, "max_fail_streak": args.max_fail_streak,
        "min_nodes_before_finish": args.min_nodes_before_finish,
        "max_premature_retries": args.max_premature_retries,
        "max_citation_retries": args.max_citation_retries,
        "max_redundant": args.max_redundant,
        "require_citation": args.require_citation,
        "topk": args.topk, "max_queries": args.max_queries, "max_read": args.max_read,
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty_files": len(_git("status", "--porcelain").splitlines()),
        "started_at": ts,
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 逐题执行 ----
    samples: list[dict] = []
    for i, rec in enumerate(records):
        t0 = time.time()
        _LLM_CALLS["n"] = 0
        try:
            with trace_file(out_dir / "searcher_trace.jsonl", enabled=args.searcher_trace):
                if args.arm == "mindsearch":
                    res = asyncio.run(run_mindsearch(app, rec["question"]))
                elif args.arm == "no_search":
                    res = asyncio.run(run_no_search(llm, rec["question"]))
                else:
                    res = asyncio.run(run_single_turn(searcher, llm, rec["question"]))
            err = ""
        except Exception as exc:                     # 单题失败不打断整批，但要留痕
            res = {"answer": "", "answer_raw": "", "dag": {}, "turns": 0,
                   "truncated": True, "stop_reason": "exception", "format_ok": False,
                   "citation_ok": False, "citation_unknown": [],
                   "prompt_tokens": 0, "completion_tokens": 0}
            err = f"{type(exc).__name__}; details redacted"

        latency = round(time.time() - t0, 3)
        pred = (res["answer"] or "").strip()
        has_gold = rec["gold"] not in (None, "")
        score = M.score(rec["kind"], pred, rec["gold"]) if has_gold else None
        case_id = str(i)

        write_case_artifacts(out_dir, case_id, res["dag"])
        write_sources(out_dir, case_id, res["dag"])

        sample = {
            "case_id": case_id, "task": rec["task"], "kind": rec["kind"],
            "scorer_kind": rec["kind"], "method": args.arm,
            "arm": args.arm, "question": rec["question"], "gold": rec["gold"],
            "prediction": pred, "prediction_raw": res["answer_raw"],
            "score": score,
            "is_correct": (bool(score == 1.0) if score is not None else None),
            "format_ok": res["format_ok"], "stop_reason": res["stop_reason"],
            "citation_ok": res.get("citation_ok"),
            "citation_unknown": res.get("citation_unknown") or [],
            "turns": res["turns"], "truncated": res["truncated"],
            "n_nodes": len(res["dag"].get("nodes", [])),
            "dag_depth": dag_depth(res["dag"]),
            "prompt_tokens": res["prompt_tokens"],
            "completion_tokens": res["completion_tokens"],
            **{key: int(res.get(key, 0) or 0) for key in COST_FIELDS},
            # 框架 aggregate_samples 认的规范字段名（别名，避免 token 被算成 0）
            "input_positions_total": res["prompt_tokens"],
            "output_tokens_total": res["completion_tokens"],
            "num_model_calls": int(_LLM_CALLS["n"]),
            "case_wall_time_s": latency,
            "latency_s": latency, "error": err,
        }
        samples.append(sample)
        with (out_dir / "predictions.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
        shown = "n/a" if score is None else f"{score:.0f}"
        print(f"[{i + 1}/{len(records)}] score={shown} stop={res['stop_reason']} "
              f"cit={res.get('citation_ok')} pred={pred[:40]!r} err={err[:60]}", flush=True)

    summary = summarize(samples)
    framework = M.aggregate_samples(
        samples,
        run_info={"model": args.model, "method": args.arm,
                  "task": args.benchmark or "adhoc",
                  "scorer_kind": records[0]["kind"] if records else "exact",
                  "git_sha": config["git_sha"]})
    (out_dir / "metrics.json").write_text(
        json.dumps({**summary, "framework": framework}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"[{args.arm}] accuracy={summary['accuracy']} "
          f"mean_score={framework.get('mean_score')} out_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()
