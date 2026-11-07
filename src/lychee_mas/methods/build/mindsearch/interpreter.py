"""规划指令（ops）的解析与应用——planner 的"函数调用层"。

LLM 输出一段 JSON：{"nodes":[{"query":..,"parents":[..]}], "edges":[[src,dst]], "finish":..}
本模块负责：宽松提取 → 严格校验（失败给出可回灌的反馈） → 应用到 TaskDAG。
失败一律显式抛错，由 plan_with_retry 决定是否重试；绝不静默兜底。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .dag import TaskDAG

OPS_KEYS = {"nodes", "edges", "finish", "response_parents"}


class OpsParseError(ValueError):
    """JSON 结构/字段不合法（可回灌给 LLM 重试）。"""


class OpsApplyError(ValueError):
    """引用了不存在的节点或成环（可回灌给 LLM 重试）。"""


@dataclass
class NewNode:
    query: str
    parents: list[str] = field(default_factory=list)


@dataclass
class Ops:
    nodes: list[NewNode] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)
    finish: str | None = None
    response_parents: list[str] = field(default_factory=list)   # finish 时响应节点的依赖
    truncated: bool = False
    warnings: list[str] = field(default_factory=list)


def _extract_json(text: str) -> str:
    """从带前后缀/```fence 的输出里取出第一个平衡的 JSON 对象。"""
    start = text.find("{")
    if start < 0:
        raise OpsParseError("No JSON object found in the output (must start with '{')")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise OpsParseError("Unbalanced JSON braces (missing '}')")


def parse_ops(text: str) -> Ops:
    try:
        data = json.loads(_extract_json(text))
    except json.JSONDecodeError as exc:
        raise OpsParseError(f"JSON syntax error: {exc.msg} (at position {exc.pos})") from exc
    if not isinstance(data, dict):
        raise OpsParseError(f"Top-level value must be a JSON object, got {type(data).__name__}")

    if "parents" in data and "response_parents" not in data:
        raise OpsParseError(
            "Top-level field renamed to response_parents "
            "(node dependencies go in nodes[i].parents)")
    unknown = set(data) - OPS_KEYS
    if unknown:
        raise OpsParseError(f"Unknown field(s) {sorted(unknown)}; allowed: {sorted(OPS_KEYS)}")

    raw_nodes = data.get("nodes", [])
    if not isinstance(raw_nodes, list):
        raise OpsParseError("nodes must be a list")
    nodes: list[NewNode] = []
    for i, item in enumerate(raw_nodes):
        if not isinstance(item, dict):
            raise OpsParseError(f"nodes[{i}] must be an object")
        query = item.get("query")
        if not isinstance(query, str) or not query.strip():
            raise OpsParseError(f"nodes[{i}].query must be a non-empty string")
        parents = item.get("parents", [])
        if not isinstance(parents, list) or not all(isinstance(p, str) for p in parents):
            raise OpsParseError(f"nodes[{i}].parents must be a list of strings")
        nodes.append(NewNode(query=query.strip(), parents=list(parents)))

    raw_edges = data.get("edges", [])
    if not isinstance(raw_edges, list):
        raise OpsParseError("edges must be a list")
    edges: list[tuple[str, str]] = []
    for i, item in enumerate(raw_edges):
        if (not isinstance(item, list) or len(item) != 2
                or not all(isinstance(x, str) for x in item)):
            raise OpsParseError(f"edges[{i}] must be [src, dst] with two strings")
        edges.append((item[0], item[1]))

    finish = data.get("finish")
    if finish == "":
        finish = None
    if finish is not None and (not isinstance(finish, str) or not finish.strip()):
        raise OpsParseError("finish must be a non-empty string, or omitted")

    raw_parents = data.get("response_parents", [])
    if not isinstance(raw_parents, list) or not all(isinstance(p, str) for p in raw_parents):
        raise OpsParseError("response_parents must be a list of strings")
    response_parents = list(dict.fromkeys(raw_parents))     # 去重保序

    warnings: list[str] = []
    if nodes and finish is not None:
        # 模型常见写法：本轮既扩图又给出（过早的）答案。保留扩图、丢弃 finish，
        # 否则一次"贪心回答"会把合法的 nodes 一起丢掉，导致零检索收敛。
        warnings.append("nodes and finish both present: kept nodes, ignored finish")
        finish = None
        response_parents = []
    if not nodes and finish is None:
        raise OpsParseError("Neither nodes nor finish: an empty plan is meaningless")
    if response_parents and finish is None:
        raise OpsParseError(
            "response_parents may only be used together with finish; "
            "put expansion dependencies in nodes[i].parents")
    return Ops(nodes=nodes, edges=edges,
               finish=finish.strip() if finish is not None else None,
               response_parents=response_parents,
               warnings=warnings)


def _norm_q(query: str) -> str:
    """子问题去重用的归一化：小写、压空白。"""
    return " ".join((query or "").lower().split())


def apply_ops(dag: TaskDAG, ops: Ops, *, max_subq_per_turn: int = 3,
              dedupe: bool = True) -> tuple[list[str], bool]:
    """把 ops 写进 DAG；返回 (新建节点 id, 是否因超限被截断)。

    dedupe=True 时，与已有节点（含本批已建节点）归一化后完全相同的子问题被跳过，
    记入 ops.warnings；一个都没建时返回空列表，由 planner 判定"无进展"。
    """
    if ops.finish is not None:
        return [], False

    truncated = False
    picked = ops.nodes
    if len(picked) > max_subq_per_turn:
        picked, truncated = picked[:max_subq_per_turn], True

    existing = {_norm_q(n.query) for n in dag.nodes.values()}
    created: list[str] = []
    skipped: list[str] = []
    for node in picked:
        key = _norm_q(node.query)
        if dedupe and key in existing:
            skipped.append(node.query)
            continue
        unknown = [p for p in node.parents if p not in dag.nodes]
        if unknown:
            raise OpsApplyError(
                f"父节点不存在 {unknown}；当前可用节点 {sorted(dag.nodes)}")
        try:
            created.append(dag.add_node(node.query, parents=node.parents))
        except ValueError as exc:
            raise OpsApplyError(str(exc)) from exc
        existing.add(key)

    if skipped:
        ops.warnings.append(f"skipped {len(skipped)} duplicate sub-question(s): {skipped[:2]}")

    for src, dst in ops.edges:
        if src not in dag.nodes or dst not in dag.nodes:
            raise OpsApplyError(
                f"edge ({src!r},{dst!r}) 引用了不存在的节点；"
                f"当前可用节点 {sorted(dag.nodes)}")
        try:
            dag.add_edge(src, dst)
        except ValueError as exc:
            raise OpsApplyError(str(exc)) from exc
    return created, truncated


def _text(result: Any) -> str:
    """兼容 LLM 后端返回对象（带 .text）或裸字符串。"""
    text = getattr(result, "text", result)
    if not isinstance(text, str):
        raise TypeError(f"规划 LLM 必须返回 str 或带 .text 的对象，得到 {type(result).__name__}")
    return text


async def plan_with_retry(llm: Callable[[str], Awaitable[Any]],
                          prompt_builder: Callable[[str], str],
                          dag: TaskDAG, *, max_retry: int = 2,
                          max_subq_per_turn: int = 3) -> tuple[Ops, list[str], bool]:
    """调用 LLM 产出 ops；解析/应用失败把错误回灌重试，超限则显式抛错。"""
    feedback = ""
    last: Exception | None = None
    for _ in range(max_retry + 1):
        text = _text(await llm(prompt_builder(feedback)))
        import os
        if os.environ.get("LYCHEE_DEBUG_PLANNER") == "1":          # ← 新增
            print(f"\n{text}\n----- end -----",
                  flush=True)
        try:
            ops = parse_ops(text)
            created, truncated = apply_ops(dag, ops, max_subq_per_turn=max_subq_per_turn)
            return ops, created, truncated
        except (OpsParseError, OpsApplyError) as exc:
            last = exc
            feedback = f"{exc}. Output only the corrected JSON, without explanation."
    raise OpsParseError(f"Planning failed {max_retry + 1} times in a row; gave up: {last}")
