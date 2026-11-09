import asyncio

import pytest

from lychee_mas.methods.build.mindsearch.dag import TaskDAG
from lychee_mas.methods.build.mindsearch.interpreter import (
    OpsApplyError, OpsParseError, apply_ops, parse_ops, plan_with_retry)


def test_parse_with_prose_and_fence():
    text = ('thinking...\n```json\n'
            '{"nodes":[{"query":"sub1","parents":[]},'
            '{"query":"sub2","parents":["n0"]}],"edges":[["n0","n1"]]}\n```\n')
    ops = parse_ops(text)
    assert len(ops.nodes) == 2 and ops.edges == [("n0", "n1")]


@pytest.mark.parametrize("bad", [
    "no json",
    '{"nodes":[]}',
    '{"nodes":[],"foo":1}',
    '{"nodes":[{"query":""}]}',
])
def test_illegal_ops_raise(bad):
    with pytest.raises(OpsParseError):
        parse_ops(bad)


def test_nodes_with_finish_keeps_nodes():
    """模型常见的"既扩图又回答"：保留 nodes、丢弃过早的 finish（v1.5 起）。"""
    ops = parse_ops('{"nodes":[{"query":"q"}],"finish":"ans"}')
    assert [n.query for n in ops.nodes] == ["q"]
    assert ops.finish is None
    assert any("忽略 finish" in w for w in ops.warnings)


def test_apply_unknown_parent_raises():
    dag = TaskDAG("q")
    ops = parse_ops('{"nodes":[{"query":"q","parents":["n9"]}]}')
    with pytest.raises(OpsApplyError, match="n9"):
        apply_ops(dag, ops)


def test_truncation_flag():
    dag = TaskDAG("q")
    ops = parse_ops('{"nodes":[{"query":"a"},{"query":"b"},{"query":"c"},{"query":"d"}]}')
    created, truncated = apply_ops(dag, ops, max_subq_per_turn=3)
    assert len(created) == 3 and truncated is True


def test_retry_then_success():
    calls = {"n": 0}

    class R:
        def __init__(self, t):
            self.text = t

    async def llm(prompt):
        calls["n"] += 1
        return R("not json" if calls["n"] == 1 else '{"nodes":[{"query":"subX"}]}')

    _, created, _ = asyncio.run(plan_with_retry(llm, lambda fb: "P" + fb, TaskDAG("q")))
    assert len(created) == 1 and calls["n"] == 2


def test_retry_exhausted_raises():
    async def bad_llm(prompt):
        return "never json"

    with pytest.raises(OpsParseError, match="放弃"):
        asyncio.run(plan_with_retry(bad_llm, lambda fb: "P", TaskDAG("q"), max_retry=1))

