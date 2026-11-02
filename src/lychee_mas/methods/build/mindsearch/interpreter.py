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

OPS_KEYS = {"nodes", "edges", "finish"}


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
    truncated: bool = False


def _extract_json(text: str) -> str:
    """从带前后缀/```fence 的输出里取出第一个平衡的 JSON 对象。"""
    start = text.find("{")
    if start < 0:
        raise OpsParseError("输出里找不到 JSON 对象（必须以 { 开头）")
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
    raise OpsParseError("JSON 对象括号不闭合（缺少 }）")


def parse_ops(text: str) -> Ops:
    try:
        data = json.loads(_extract_json(text))
    except json.JSONDecodeError as exc:
        raise OpsParseError(f"JSON 语法错误：{exc.msg}（位置 {exc.pos}）") from exc
    if not isinstance(data, dict):
        raise OpsParseError(f"顶层必须是 JSON 对象，得到 {type(data).__name__}")

    unknown = set(data) - OPS_KEYS
    if unknown:
        raise OpsParseError(f"出现未知字段 {sorted(unknown)}；只允许 {sorted(OPS_KEYS)}")

    raw_nodes = data.get("nodes", [])
    if not isinstance(raw_nodes, list):
        raise OpsParseError("nodes 必须是数组")
    nodes: list[NewNode] = []
    for i, item in enumerate(raw_nodes):
        if not isinstance(item, dict):
            raise OpsParseError(f"nodes[{i}] 必须是对象")
        query = item.get("query")
        if not isinstance(query, str) or not query.strip():
            raise OpsParseError(f"nodes[{i}].query 必须是非空字符串")
        parents = item.get("parents", [])
        if not isinstance(parents, list) or not all(isinstance(p, str) for p in parents):
            raise OpsParseError(f"nodes[{i}].parents 必须是字符串数组")
        nodes.append(NewNode(query=query.strip(), parents=list(parents)))

    raw_edges = data.get("edges", [])
    if not isinstance(raw_edges, list):
        raise OpsParseError("edges 必须是数组")
    edges: list[tuple[str, str]] = []
    for i, item in enumerate(raw_edges):
        if (not isinstance(item, list) or len(item) != 2
                or not all(isinstance(x, str) for x in item)):
            raise OpsParseError(f"edges[{i}] 必须是 [src, dst] 两个字符串")
        edges.append((item[0], item[1]))

    finish = data.get("finish")
    if finish is not None and (not isinstance(finish, str) or not finish.strip()):
        raise OpsParseError("finish 必须是非空字符串或省略")

    if nodes and finish is not None:
        raise OpsParseError("同一轮不能既新增子问题又 finish：二选一")
    if not nodes and finish is None:
        raise OpsParseError("既没有 nodes 也没有 finish：空规划无意义")
    return Ops(nodes=nodes, edges=edges,
               finish=finish.strip() if finish is not None else None)


def apply_ops(dag: TaskDAG, ops: Ops, *, max_subq_per_turn: int = 3) -> tuple[list[str], bool]:
    """把 ops 写进 DAG；返回 (新建节点 id, 是否因超限被截断)。"""
    if ops.finish is not None:
        return [], False

    truncated = False
    picked = ops.nodes
    if len(picked) > max_subq_per_turn:
        picked, truncated = picked[:max_subq_per_turn], True

    created: list[str] = []
    for node in picked:
        unknown = [p for p in node.parents if p not in dag.nodes]
        if unknown:
            raise OpsApplyError(
                f"父节点不存在 {unknown}；当前可用节点 {sorted(dag.nodes)}")
        try:
            created.append(dag.add_node(node.query, parents=node.parents))
        except ValueError as exc:
            raise OpsApplyError(str(exc)) from exc

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
        try:
            ops = parse_ops(text)
            created, truncated = apply_ops(dag, ops, max_subq_per_turn=max_subq_per_turn)
            return ops, created, truncated
        except (OpsParseError, OpsApplyError) as exc:
            last = exc
            feedback = f"上一次输出不合法：{exc}。请只输出修正后的 JSON，不要解释。"
    raise OpsParseError(f"规划连续 {max_retry + 1} 次不合法，已放弃：{last}")
