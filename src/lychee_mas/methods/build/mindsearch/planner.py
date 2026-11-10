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
from .citation import check_citations
from .dag import TaskDAG
from .interpreter import OpsApplyError, OpsParseError, plan_with_retry

# ---- 工程常量（非论文超参）：只控制 prompt 体积，不改变判定逻辑 ----
_SUMMARY_LIMIT = 200      # 摘要里每条结论的最大字符数
_PARENT_LIMIT = 800       # 单条父结论进入 context 的上限（中文≈800 token）
_ROWS_MARGIN = 8          # 摘要行数 = max_nodes + 8，正常不会触发截断
_CONTEXT_BUDGET = 2400    # 单节点 context 的总字符预算（多父节点时均分）


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


class _SafeDict(dict):
    """str.format_map 用：未提供的占位符原样保留，避免 KeyError 打断规划。"""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _render_system(system_prompt: str, question: str) -> str:
    """把 spec.system_prompt 渲染成一段可用的前置指令（支持 {question}/{context}）。

    这是与 prerun/MASPO 的接缝：优化器改写的正是 spec.system_prompt，
    必须真正进入每轮提示，否则改写不生效。
    """
    text = (system_prompt or "").strip()
    if not text:
        return ""
    try:
        return text.format_map(_SafeDict(question=question, context="(see the task graph below)",
                                         dag="", memory=""))
    except (KeyError, ValueError, IndexError):
        return text


# ---------------------------------------------------------------- 上下文

def build_context(dag: TaskDAG, qid: str, *, parent_limit: int = _PARENT_LIMIT,
                  budget: int = _CONTEXT_BUDGET) -> str:
    """检索者上下文：父节点结论（可多条）+ 分隔线 + 根问题（不含兄弟/历史）。

    多父节点时按总字符预算均分，避免 5 个以上父节点把 prompt 撑爆。
    """
    if qid not in dag.nodes:
        raise ValueError(f"build_context: unknown node {qid!r}")
    node = dag.nodes[qid]
    per = min(parent_limit, max(200, budget // max(1, len(node.parents))))
    parents = [f"[{p}] {_clip(dag.nodes[p].answer, per)}" for p in node.parents]
    head = "\n".join(parents) if parents else "(no parent: this node maps to the root question)"
    return f"{head}\n---\nRoot question: {dag.root_query}"


# ---------------------------------------------------------------- 摘要

def _src_url(src: Any) -> str:
    if isinstance(src, dict):
        return str(src.get("url") or "")
    return str(getattr(src, "url", "") or "")


def _blocked_by(dag: TaskDAG, qid: str) -> str:
    """该节点若有 failed 祖先，返回该祖先 id；否则返回 ""（多父节点时最易发生）。"""
    stack, seen = list(dag.nodes[qid].parents), set()
    while stack:
        cur = stack.pop()
        if cur in seen or cur not in dag.nodes:
            continue
        seen.add(cur)
        if dag.nodes[cur].status == "failed":
            return cur
        stack.extend(dag.nodes[cur].parents)
    return ""


def _summarize(dag: TaskDAG, *, limit: int = _SUMMARY_LIMIT, max_rows: int = 48) -> str:
    """给规划者看：id、状态（失败标注次数/被阻塞）、子问题、结论摘要、来源 URL。

    来源必须出现在提示里：否则规划者无法引用检索结果，citation 校验永远不通过。
    """
    fails = Counter(_norm_q(n.query) for n in dag.nodes.values() if n.status == "failed")
    order = dag.topological()
    rows: list[str] = []
    for qid in order[:max_rows]:
        n = dag.nodes[qid]
        tries = fails.get(_norm_q(n.query), 0)
        status = f"{n.status}×{tries}" if tries > 1 else n.status
        if n.status == "pending":
            blocker = _blocked_by(dag, qid)
            if blocker:
                status = f"blocked by {blocker}"
        srcs = [u for u in (_src_url(s) for s in (n.sources or [])) if u][:2]
        src_txt = f" [src: {', '.join(srcs)}]" if srcs else ""
        rows.append(f"- {qid} [{status}] {_clip(n.query, 80)} => "
                    f"{_clip(n.answer, limit)}{src_txt}")
    if not rows:
        return "(empty)"
    if len(order) > max_rows:
        rows.append(f"- … ({len(order) - max_rows} more nodes not shown)")
    return "\n".join(rows)


# ---------------------------------------------------------------- 提示词

# _OPS_EXAMPLE = (
#     '{"nodes": [{"query": "<子问题>", "parents": ["<已有id>"]}],\n'
#     ' "edges": [["<src>", "<dst>"]],\n'
#     ' "finish": "Answer: <短答案>\\n<解释与来源>"}'
# )
_EXPAND_EXAMPLE = '{"nodes": [{"query": "<sub-question>", "parents": ["<existing-id>"]}]}'
_FINISH_EXAMPLE = ('{"finish": "Answer: <short answer>\\n<explanation and sources>", '
                   '"response_parents": ["<completed-id>"]}')


def build_plan_prompt(question: str, dag: TaskDAG, feedback: str = "", *,
                      finish_only: bool = False, give_up: bool = False,
                      max_rows: int = 48, must_expand: bool = False,
                      system_prompt: str = "") -> str:
    head: list[str] = []
    sys_text = _render_system(system_prompt, question)
    if sys_text:
        head += [sys_text, ""]
    else:
        head.append(f"Question: {question}")
    head += [
        "Current task graph:",
        _summarize(dag, max_rows=max_rows),
        f"Existing node ids: {sorted(dag.nodes)}",
    ]
    if give_up:
        return "\n".join(head + [
            "Repeated searches produced no usable information; do not search again.",
            'Output exactly one JSON object: {"finish": "Answer: <what can be confirmed, '
            'or cannot confirm>\\nMissing: <what is missing and which angles were tried>"}',
            "Replace every angle-bracket placeholder with real content; keep no placeholders.",
        ])
    if finish_only:
        return "\n".join(head + [
            "The turn/budget limit is reached; do not expand the graph any more.",
            'Output exactly one JSON object: {"finish": "Answer: <short answer>\\n'
            '<explanation and sources>"}',
            "Replace every angle-bracket placeholder with real content; keep no placeholders.",
        ])
    if must_expand:
        # 首轮（还没有任何检索结果）只给扩图模板：让 finish 在结构上无处可填
        return "\n".join(head + [
            "This turn you may only expand the graph: output the JSON below and no finish.",
            _EXPAND_EXAMPLE,
            "At most 3 sub-questions per turn; parents must be existing node ids "
            "([] for first-level sub-questions); do not repeat existing sub-questions; "
            "omit any field you do not use.",
        ])
    if feedback:
        head.append(f"Your previous output was invalid: {feedback}")
    head += [
        "Output exactly one JSON object per turn, in one of the two forms below "
        "(no explanation, no markdown code fence):",
        "A. Expand the graph (no answer this turn):",
        _EXPAND_EXAMPLE,
        "B. Converge (no expansion this turn):",
        _FINISH_EXAMPLE,
        "Rules:",
        "1. Use either the nodes field or the finish field -- never both, never neither;",
        "2. At most 3 sub-questions per turn;",
        "3. parents must be existing node ids (e.g. n0, n1); use [] for first-level sub-questions;",
        "4. Do not repeat an existing sub-question;",
        "5. A sub-question marked [failed] may only be retried with different wording;",
        "6. If information is insufficient, expand the graph (nodes); if it is sufficient, "
        "output finish only, with source links;",
        "7. The first line of finish must be `Answer: <short answer>` "
        "(entity / yes / no / number), with the explanation on the following lines;",
        "8. finish must include response_parents listing only the completed node ids you actually "
        "used (e.g. [\"n1\",\"n3\"]); the harness builds the [FINAL RESPONSE] node from them.",
        "Replace every angle-bracket placeholder with real content; keep no placeholders.",
    ]
    return "\n".join(head)


# ---------------------------------------------------------------- 记账

def _counted(llm: Callable[[str], Any], tok: dict) -> Callable[[str], Any]:
    async def inner(prompt: str):
        res = await llm(prompt)
        tok["p"] += int(getattr(res, "n_prompt_pos", 0) or 0)
        tok["c"] += int(getattr(res, "n_gen_tokens", 0) or 0)
        tok["planner_prompt_tokens"] = tok.get("planner_prompt_tokens", 0) + int(
            getattr(res, "n_prompt_pos", 0) or 0)
        tok["planner_completion_tokens"] = tok.get("planner_completion_tokens", 0) + int(
            getattr(res, "n_gen_tokens", 0) or 0)
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
            for key, value in (
                ("searcher_prompt_tokens", getattr(res, "prompt_tokens", 0)),
                ("searcher_completion_tokens", getattr(res, "completion_tokens", 0)),
                ("queries_count", len(getattr(res, "queries", None) or [])),
                ("pages_read", getattr(res, "pages_read", 0)),
            ):
                tok[key] = tok.get(key, 0) + int(value or 0)
            if getattr(res, "is_ok", lambda: False)():
                dag.mark_done(qid, res.answer, [s.to_dict() for s in res.sources])
                ok += 1
            elif getattr(res, "status", "") == "empty":
                dag.mark_done(qid, "(no relevant information found)")   # 空结论不阻塞子节点
                empty += 1
            else:
                dag.mark_failed(qid, getattr(res, "error", "") or "searcher error")
                err += 1

    await asyncio.gather(*(one(q) for q in ready))
    return ok, empty, err


# ---------------------------------------------------------------- 收尾

RESPONSE_NODE = "[FINAL RESPONSE]"   # 语义等同论文的 response 节点，沿用既有命名

def _final(state: dict, name: str, dag: TaskDAG, text: str, turn: int, tok: dict,
           truncated: bool, reason: str, add_response: bool = True,
           citation_ok: bool = False,
           citation_unknown: list[str] | None = None,
           response_parents: list[str] | None = None) -> dict:
    if reason != "finish" and "[UNRESOLVED]" not in text:
        # 非正常收敛的答案必须带标注，避免被评估当成有效输出
        text = f"{text}\n\n[UNRESOLVED] stop_reason={reason}; turns={turn}; nodes={len(dag)}"
    fmt = format_answer(text)
    if add_response and not any(n.query == RESPONSE_NODE for n in dag.nodes.values()):
        done_ids = [q for q in dag.topological() if dag.nodes[q].status == "done"]
        # 论文：response 节点由模型指定依赖；未指定时退回"连接全部已完成节点"
        deps = [p for p in (response_parents or done_ids) if p in dag.nodes]
        # 响应节点汇总这些依赖节点的来源，便于审计（答案引用 vs 图内来源）
        seen: set[str] = set()
        sources: list[dict] = []
        for q in deps:
            for src in dag.nodes[q].sources or []:
                url = _src_url(src)
                if url and url not in seen:
                    seen.add(url)
                    sources.append({"url": url,
                                    "title": str((src.get("title") if isinstance(src, dict)
                                                  else getattr(src, "title", "")) or ""),
                                    "snippet": ""})
        rid = dag.add_node(RESPONSE_NODE, parents=deps)
        dag.mark_done(rid, fmt.long, sources)
    return {
        "outputs": {**state.get("outputs", {}), name: fmt.long},
        "dag": dag.to_dict(),
        "answer": fmt.short,
        "format_ok": fmt.format_ok,
        "turns": turn,
        "truncated": truncated,
        "stop_reason": reason,
        "citation_ok": citation_ok,
        "citation_unknown": list(citation_unknown or []),
        "prompt_tokens": tok["p"],
        "completion_tokens": tok["c"],
        **{key: tok.get(key, 0) for key in (
            "planner_prompt_tokens", "planner_completion_tokens",
            "searcher_prompt_tokens", "searcher_completion_tokens", "queries_count", "pages_read")},
    }


def _fallback(dag: TaskDAG, why: str) -> str:
    done = [dag.nodes[q] for q in dag.topological()
            if dag.nodes[q].status == "done" and dag.nodes[q].answer]
    tip = _clip(done[-1].answer, 120) if done else "(no usable conclusion collected)"
    return f"Answer: {tip}\n[UNRESOLVED] {why}; {len(dag)} nodes in total."


# ---------------------------------------------------------------- 节点工厂

def make_planner_node(spec: Any, is_terminal: bool, *, searcher: Any, llm: Any,
                      max_turns: int = 10, max_subq_per_turn: int = 3,
                      max_nodes: int = 40, max_concurrency: int = 8,
                      wall_clock_s: float = 300.0, max_retry: int = 2,
                      max_fail_streak: int = 2, max_failed_nodes: int = 5,
                      add_response_node: bool = True,
                      min_nodes_before_finish: int = 1,
                      max_premature_retries: int = 2,
                      max_citation_retries: int = 1,
                      max_redundant: int = 2,
                      max_response_retries: int = 1,
                      require_citation: bool = True):
    if not is_terminal:
        raise ValueError("mindsearch has a single planner node; is_terminal must be True")

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
        for key in ("planner_prompt_tokens", "planner_completion_tokens",
                    "searcher_prompt_tokens", "searcher_completion_tokens",
                    "queries_count", "pages_read"):
            tok[key] = int(state.get(key, 0) or 0)
        truncated, stop = False, "max_turns"
        fail_streak = 0
        stalled = 0
        premature = 0            # 连续"未检索就 finish"的次数
        citation_retry = 0       # 引用不合规的重答次数
        redundant = 0            # 连续"只提重复子问题"的轮数
        response_retry = 0       # response 依赖非法时的重答次数
        pending_feedback = ""    # 待回灌给下一次规划调用的提示（跨轮传递）
        citation_ok, citation_unknown = False, []
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

            must_expand = not any(n.status == "done" for n in dag.nodes.values())
            try:
                ops, created, tr = await plan_with_retry(
                    _counted(llm, tok),
                    lambda fb: build_plan_prompt(
                        question, dag, fb or pending_feedback,
                        max_rows=max_rows, must_expand=must_expand,
                        system_prompt=spec.system_prompt),
                    dag, max_retry=max_retry, max_subq_per_turn=max_subq_per_turn)
            except (OpsParseError, OpsApplyError) as exc:
                return _final(state, spec.name, dag, _fallback(dag, f"planning failed: {exc}"),
                              turn, tok, True, "plan_error", add_response_node)
            pending_feedback = ""

            if not created and ops.finish is None:
                # 全是重复子问题：不推进 DAG，提醒换角度；连续无新节点则收敛
                redundant += 1
                if redundant >= max_redundant:
                    truncated, stop = True, "redundant"
                    break
                pending_feedback = ("These sub-questions already exist (or were deduplicated). "
                                    "Propose a new angle, or output finish.")
                continue
            redundant = 0

            truncated = truncated or tr
            if ops.finish is not None:
                done = [n for n in dag.nodes.values() if n.status == "done"]
                if len(done) < min_nodes_before_finish:
                    # finish 是"终态动作"：还没有任何检索结果时不予接受
                    premature += 1
                    if premature <= max_premature_retries:
                        pending_feedback = (
                            "No search result has been collected yet; do not answer directly. "
                            "Output only nodes (sub-questions).")
                        continue
                    # 连续拒绝仍不分解：程序化兜底，保证至少发生一次检索
                    truncated, stop = True, "forced_expand"
                    if not dag.nodes:
                        forced = dag.add_node(question, parents=[])
                        pending_feedback = (
                            f"Forced the root question into node {forced} and searched it. "
                            "Next turn decide expand or finish based on the results.")
                        continue
                    break

                report = check_citations(ops.finish, dag, require_citation=require_citation)
                citation_ok, citation_unknown = report.ok, report.unknown
                if not report.ok and citation_retry < max_citation_retries:
                    citation_retry += 1
                    pending_feedback = report.feedback()
                    continue

                # 论文：response 节点由模型指定依赖；非法依赖先回灌一次，再退回自动连接
                deps = ops.response_parents
                bad = [p for p in deps
                       if p not in dag.nodes or dag.nodes[p].status != "done"]
                if bad and response_retry < max_response_retries:
                    response_retry += 1
                    pending_feedback = (
                        f"[FINAL RESPONSE] may only link to completed nodes; invalid: {bad}. "
                        "Re-emit finish with response_parents listing completed ids only.")
                    continue
                if bad:
                    deps = []
                return _final(state, spec.name, dag, ops.finish, turn, tok,
                              truncated, "finish", add_response_node,
                              citation_ok=citation_ok,
                              citation_unknown=citation_unknown,
                              response_parents=deps)
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
            text = ops.finish or _fallback(dag, f"did not converge within {turn} turns ({stop})")
        except (OpsParseError, OpsApplyError):
            text = _fallback(dag, f"did not converge within {turn} turns ({stop})")
        # 收尾时只记录引用状态、不再回灌（已无预算）；但"没引用"必须如实记 False，
        # 否则 require_citation=False 会让它被算成合规。
        report = check_citations(text, dag, require_citation=True)
        return _final(state, spec.name, dag, text, turn, tok, True, stop,
                      add_response_node, citation_ok=report.ok,
                      citation_unknown=report.unknown)

    return node


def planner_node_factory(*, searcher: Any, llm: Any, **kwargs: Any):
    """返回 node_factory(spec, is_terminal)，供 build_langgraph 使用。"""
    return lambda spec, is_terminal: make_planner_node(
        spec, is_terminal, searcher=searcher, llm=llm, **kwargs)
