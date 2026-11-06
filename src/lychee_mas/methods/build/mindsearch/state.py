"""MindSearch 的 LangGraph 状态形状（默认值；实验脚本可注入同形状的自定义 schema）。"""
from __future__ import annotations

from typing import Any, Dict, TypedDict


class MindSearchState(TypedDict, total=False):
    question: str                  # 根问题
    outputs: Dict[str, str]        # 节点名 -> 该节点最终文本（终端节点即答案）
    answer: str        # 短答案（打分口径；由 planner 格式化产出）
    format_ok: bool    # finish 是否带规范的 Answer: 行（prompt 遵从度）
    dag: Dict[str, Any]            # TaskDAG.to_dict()（落盘 dag/<qid>.json）
    turns: int                     # 已执行的规划轮数
    truncated: bool                # 是否因上限被截断（轮数/子问题数/节点数）
    prompt_tokens: int
    completion_tokens: int
    error: str                     # 非空表示本次运行失败（显式记录，不静默）
    stop_reason: str               # finish|giveup|max_turns|max_nodes|timeout|stalled|plan_error
                                   # |redundant|forced_expand|premature_finish|exception
    citation_ok: bool              # 答案引用是否可追溯到检索 sources（引用级校验）
    citation_unknown: list[str]   # 未被检索到的引用 URL（citation_ok=False 时非空）
