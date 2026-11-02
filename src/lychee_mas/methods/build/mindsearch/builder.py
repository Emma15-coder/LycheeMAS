"""graph_builder/mindsearch —— 把 MindSearch 的协作结构映射到契约 StateGraph。

设计取舍：
- LangGraph 侧只放静态骨架 START→planner→END；
- 子问题 DAG 在 planner 节点内部逐轮增长，落盘到 state["dag"]；
- 因此不违反 graphview「静态 DAG + 唯一终端」的契约，prerun（MASPO/剪枝）照常可用。
"""
from __future__ import annotations

from typing import Any, Callable

from ....core.registry import REGISTRY
from ....core.types import AgentSpec
from .prompts import PLANNER_PROMPT

NodeFactory = Callable[[AgentSpec, bool], Any]


@REGISTRY.register("graph_builder", "mindsearch")
class MindSearchGraphBuilder:
    name = "mindsearch"

    def __init__(self, node_factory: NodeFactory, state_schema: Any, *,
                 planner_prompt: str = PLANNER_PROMPT,
                 planner_name: str = "planner") -> None:
        if node_factory is None or state_schema is None:
            raise ValueError("graph_builder/mindsearch 需要 node_factory 与 state_schema"
                             "（节点语义与状态形状由实验方定义）")
        if not callable(node_factory):
            raise TypeError(f"node_factory 必须可调用，得到 {type(node_factory).__name__}")
        self.node_factory = node_factory
        self.state_schema = state_schema
        self.planner_prompt = planner_prompt
        self.planner_name = planner_name

    def build(self) -> Any:
        from langgraph.graph import END, START, StateGraph  # 惰性：selfcheck 必须 NONE

        spec = AgentSpec(name=self.planner_name, role="planner",
                         system_prompt=self.planner_prompt, meta={"predecessors": []})
        sg = StateGraph(self.state_schema)
        sg.add_node(spec.name, self.node_factory(spec, True),
                    metadata={"agent_spec": spec})
        sg.add_edge(START, spec.name)
        sg.add_edge(spec.name, END)
        return sg
