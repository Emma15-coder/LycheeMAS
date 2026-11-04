"""planner 节点：MindSearch 的规划循环（build 接缝）。

论文对应：
- §2.1 规划即建图：每轮把问题拆成子问题并入 DAG（JSON ops 代替原文的代码动作）
- §2.1 并行：互不依赖的子问题并发检索
- §2.3 上下文：检索者只看「父节点结论 + 根问题」
- 附录 E.2 / Figure 9：多次检索无果后放弃该问题，直接生成响应节点

调度语义（本项目实验设计）：
- TaskDAG.ready() 只把 done 当终态；failed 会永久阻塞其子节点（不改 ready()）
- 失败不复活子树，由规划者新建无父依赖的替代节点绕过
- 连续无果达到阈值（max_fail_streak）或失败节点过多（max_failed_nodes）→ giveup
"""
from __future__ import annotations

import asyncio
import time
from collections import Counter
from typing import Any, Callable

from .answer_format import format_answer
from .dag import TaskDAG
from .interpreter import OpsApplyError, OpsParseError, plan_with_retry

# ---- 工程常量（非论文超参）：只控制 prompt 体积，不改变判定逻辑 ----
_SUMMARY_LIMIT = 200      # 摘要里每条结论的最大字符数
_PARENT_LIMIT = 800       # 单条父结论进入 context 的上限（中文≈800 token）
_ROWS_MARGIN = 8          # 摘要行数 = max_nodes + 8，正常不会触发截断


def _clip(text: str, limit: int, head_ratio: float = 0.7) -> str:
    """压单行并头尾截断：结论常在末尾，只留开头会把它切掉。"""
    s = " ".join((text or "").split())
    if len(s) <= limit:
        return s
    head = int(limit * head_ratio)
    tail = max(0, limit - head - 1)
    return f"{s[:head]}…{s[-tail:]}" if tail else f"{s[:head]}…"


def _norm_q(query: str) -> str:
    return " ".join((query or "").lower().split())


# ---------------------------------------------------------------- 上下文

def build_context(dag: TaskDAG, qid: str, *, parent_limit: int = _PARENT_LIMIT) -> str:
    """检索者上下文：父节点结论（可多条）+ 分隔线 + 根问题（不含兄弟/历史）。"""
    if qid not in dag.nodes:
        raise ValueError(f"build_context: 节点不存在 {qid!r}")
    node = dag.nodes[qid]
    parents = [f"[{p}] {_clip(dag.nodes[p].answer, parent_limit)}" for p in node.parents]
    head = "\n".join(parents) if parents else "（无父任务：本节点直接对应根问题）"
    return f"{head}\n---\n根问题：{dag.root_query}"


# ---------------------------------------------------------------- 摘要

def _summarize(dag: TaskDAG, *, limit: int = _SUMMARY_LIMIT, max_rows: int = 48) -> str:
    """给规划者看：id、状态（失败标注次数）、子问题、结论摘要。"""
    fails = Counter(_norm_q(n.query) for n in dag.nodes.values() if n.status == "failed")
    order = dag.topological()
    rows: list[str] = []
    for qid in order[:max_rows]:
        n = dag.nodes[qid]
        tries = fails.get(_norm_q(n.query), 0)
        status = f"{n.status}×{tries}" if tries > 1 else n.status
        rows.append(f"- {qid} [{status}] {_clip(n.query, 80)} => {_clip(n.answer, limit)}")
    if not rows:
        return "(空)"
    if len(order) > max_rows:
        rows.append(f"- …（另有 {len(order) - max_rows} 个节点未展示）")
    return "\n".join(rows)


# ---------------------------------------------------------------- 提示词

_OPS_EXAMPLE = (
    '{"nodes": [{"query": "<子问题>", "parents": ["<已有id>"]}],\n'
    ' "edges": [["<src>", "<dst>"]],\n'
    ' "finish": "Answer: <短答案>\\n<解释与来源>"}'
)


def build_plan_prompt(question: str, dag: TaskDAG, feedback: str = "", *,
                      finish_only: bool = False, give_up: bool = False,
                      max_rows: int = 48) -> str:
    head = [
        f"用户问题：{question}",
        "当前任务图：",
        _summarize(dag, max_rows=max_rows),
        f"已有节点 id：{sorted(dag.nodes)}",
    ]
    if give_up:
        return "\n".join(head + [
            "多次检索均未获得有效信息，禁止再发起检索。",
            '请只输出一个 JSON：{"finish": "Answer: <能确认的结论，或 未能确认>\\n'
            '未能确认：<缺失的信息 + 已尝试过的角度>"}',
            "把尖括号占位符替换成真实内容，不要保留占位符。",
        ])
    if finish_only:
        return "\n".join(head + [
            "已到轮数/预算上限，禁止再扩图。",
            '请只输出一个 JSON：{"finish": "Answer: <短答案>\\n<解释与来源>"}',
            "把尖括号占位符替换成真实内容，不要保留占位符。",
        ])
    if feedback:
        head.append(f"上一次输出不合法：{feedback}")
    head += [
        "每轮只输出一个 JSON 对象（不要解释、不要 markdown 代码块）：",
        _OPS_EXAMPLE,
        "规则：",
        "1. nodes 与 finish 二选一，不能同时出现，也不能都为空；",
        "2. 同一轮最多 3 个子问题；",
        "3. parents 必须是已有节点 id（形如 n0、n1），第一层子问题用空数组；",
        "4. 不要重复已有子问题；",
        "5. 标 [failed] 的子问题只能换一种问法重试，不要原样重发；",
        "6. 信息不足就继续扩图（给 nodes）；信息足够就只给 finish（含来源链接）；",
        "7. finish 的第一行必须是 `Answer: <短答案>`（实体/yes/no/数字），第二行起写解释。",
        "把尖括号占位符替换成真实内容，不要保留占位符。",
    ]
    return "\n".join(head)


# ---------------------------------------------------------------- 记账

def _counted(llm: Callable[[str], Any], tok: dict) -> Callable[[str], Any]:
    async def inner(prompt: str):
        res = await llm(prompt)
        tok["p"] += int(getattr(res, "n_prompt_pos", 0) or 0)
        tok["c"] += int(getattr(res, "n_gen_tokens", 0) or 0)
        return res
    return inner


# ---------------------------------------------------------------- 检索一轮

async def _retrieve(ready: list[str], dag: TaskDAG, searcher: Any, tok: dict,
                    max_concurrency: int) -> tuple[int, int, int]:
    sem = asyncio.Semaphore(max_concurrency)
    ok = empty = err = 0

    async def one(qid: str) -> None:
        nonlocal ok, empty, err
        async with sem:
            dag.mark_running(qid)
            try:
                res = await searcher.run(dag.nodes[qid].query, build_context(dag, qid))
            except Exception as exc:                     # 契约上不该发生；兜底要留痕
                dag.mark_failed(qid, f"[exception] {type(exc).__name__}: {exc}")
                err += 1
                return
            tok["p"] += int(getattr(res, "prompt_tokens", 0) or 0)
            tok["c"] += int(getattr(res, "completion_tokens", 0) or 0)
            if getattr(res, "is_ok", lambda: False)():
                dag.mark_done(qid, res.answer, [s.to_dict() for s in res.sources])
                ok += 1
            elif getattr(res, "status", "") == "empty":
                dag.mark_done(qid, "（未检索到相关信息）")   # 空结论不阻塞子节点
                empty += 1
            else:
                dag.mark_failed(qid, getattr(res, "error", "") or "searcher error")
                err += 1

    await asyncio.gather(*(one(q) for q in ready))
    return ok, empty, err


# ---------------------------------------------------------------- 收尾

def _final(state: dict, name: str, dag: TaskDAG, text: str, turn: int, tok: dict,
           truncated: bool, reason: str, add_response: bool = True) -> dict:
    fmt = format_answer(text)
    if add_response and not any(n.query == "[FINAL RESPONSE]" for n in dag.nodes.values()):
        done_ids = [q for q in dag.topological() if dag.nodes[q].status == "done"]
        rid = dag.add_node("[FINAL RESPONSE]", parents=done_ids)
        dag.mark_done(rid, fmt.long, [])
    return {
        "outputs": {**state.get("outputs", {}), name: fmt.long},
        "dag": dag.to_dict(),
        "answer": fmt.short,
        "format_ok": fmt.format_ok,
        "turns": turn,
        "truncated": truncated,
        "stop_reason": reason,
        "prompt_tokens": tok["p"],
        "completion_tokens": tok["c"],
    }


def _fallback(dag: TaskDAG, why: str) -> str:
    done = [dag.nodes[q] for q in dag.topological()
            if dag.nodes[q].status == "done" and dag.nodes[q].answer]
    tip = _clip(done[-1].answer, 120) if done else "（未收集到有效结论）"
    return f"Answer: {tip}\n[未收敛] {why}；共 {len(dag)} 个节点。"


# ---------------------------------------------------------------- 节点工厂

def make_planner_node(spec: Any, is_terminal: bool, *, searcher: Any, llm: Any,
                      max_turns: int = 10, max_subq_per_turn: int = 3,
                      max_nodes: int = 40, max_concurrency: int = 8,
                      wall_clock_s: float = 300.0, max_retry: int = 2,
                      max_fail_streak: int = 2, max_failed_nodes: int = 5,
                      add_response_node: bool = True):
    if not is_terminal:
        raise ValueError("mindsearch 只有 planner 一个节点，is_terminal 必须为 True")

    max_rows = max_nodes + _ROWS_MARGIN          # ← 摘要行数与 max_nodes 联动

    async def node(state: dict) -> dict:
        question = state["question"]
        dag = TaskDAG.from_dict(state["dag"]) if state.get("dag") else TaskDAG(question)
        for n in dag.nodes.values():             # 上次中断的 running 归零
            if n.status == "running":
                n.status = "pending"

        turn = int(state.get("turns", 0) or 0)
        tok = {"p": int(state.get("prompt_tokens", 0) or 0),
               "c": int(state.get("completion_tokens", 0) or 0)}
        truncated, stop = False, "max_turns"
        fail_streak = 0
        stalled = 0
        t0 = time.monotonic()

        while turn < max_turns:
            if time.monotonic() - t0 > wall_clock_s:
                truncated, stop = True, "timeout"
                break
            turn += 1

            ready = dag.ready()
            if ready:
                ok, _, _ = await _retrieve(ready, dag, searcher, tok, max_concurrency)
                fail_streak = 0 if ok > 0 else fail_streak + 1
                failed_total = sum(1 for n in dag.nodes.values() if n.status == "failed")
                if fail_streak >= max_fail_streak or failed_total >= max_failed_nodes:
                    truncated, stop = True, "giveup"
                    break

            try:
                ops, created, tr = await plan_with_retry(
                    _counted(llm, tok),
                    lambda fb: build_plan_prompt(question, dag, fb, max_rows=max_rows),
                    dag, max_retry=max_retry, max_subq_per_turn=max_subq_per_turn)
            except (OpsParseError, OpsApplyError) as exc:
                return _final(state, spec.name, dag, _fallback(dag, f"规划失败：{exc}"),
                              turn, tok, True, "plan_error", add_response_node)

            truncated = truncated or tr
            if ops.finish is not None:
                return _final(state, spec.name, dag, ops.finish, turn, tok,
                              truncated, "finish", add_response_node)
            if len(dag) > max_nodes:
                truncated, stop = True, "max_nodes"
                break
            if not ready and not dag.ready():
                stalled += 1
                if stalled >= 1:
                    stop = "stalled"
                    break
            else:
                stalled = 0

        give_up = stop == "giveup"
        try:
            ops, _, _ = await plan_with_retry(
                _counted(llm, tok),
                lambda fb: build_plan_prompt(question, dag, fb, finish_only=not give_up,
                                             give_up=give_up, max_rows=max_rows),
                dag, max_retry=0)
            text = ops.finish or _fallback(dag, f"未能在 {turn} 轮内收敛（{stop}）")
        except (OpsParseError, OpsApplyError):
            text = _fallback(dag, f"未能在 {turn} 轮内收敛（{stop}）")
        return _final(state, spec.name, dag, text, turn, tok, True, stop, add_response_node)

    return node


def planner_node_factory(*, searcher: Any, llm: Any, **kwargs: Any):
    """返回 node_factory(spec, is_terminal)，供 build_langgraph 使用。"""
    return lambda spec, is_terminal: make_planner_node(
        spec, is_terminal, searcher=searcher, llm=llm, **kwargs)
