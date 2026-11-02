from typing import Any, Dict, TypedDict

import pytest

pytest.importorskip("langgraph")

from lychee_mas.core.registry import REGISTRY  # noqa: E402
from lychee_mas.plugins import build_langgraph  # noqa: E402
from lychee_mas.plugins.prerun import extract_view  # noqa: E402


class MSState(TypedDict):
    question: str
    outputs: Dict[str, str]


def _node_factory(spec, is_terminal):
    async def node(state: MSState) -> Dict[str, Any]:
        return {"outputs": {**state.get("outputs", {}), spec.name: "ANSWER: 4"}}
    return node


def test_registered():
    assert "mindsearch" in REGISTRY.snapshot()["graph_builder"]


def test_build_satisfies_contract():
    sg = build_langgraph(method="mindsearch", node_factory=_node_factory,
                         state_schema=MSState)
    view = extract_view(sg)
    assert view.names == ["planner"]
    assert view.terminal == "planner"
    assert view.predecessors == {"planner": []}


def test_missing_node_factory_raises():
    with pytest.raises(ValueError, match="node_factory"):
        build_langgraph(method="mindsearch", node_factory=None, state_schema=MSState)