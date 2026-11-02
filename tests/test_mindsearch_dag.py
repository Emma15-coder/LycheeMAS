import pytest

from lychee_mas.methods.build.mindsearch.dag import TaskDAG


def test_ready_and_status_flow():
    dag = TaskDAG("root?")
    a = dag.add_node("A")
    b = dag.add_node("B", parents=[a])
    assert dag.ready() == [a]
    dag.mark_done(a, "ans-a")
    assert dag.ready() == [b]
    assert dag.topological() == [a, b]


def test_unknown_parent_raises():
    dag = TaskDAG("root?")
    with pytest.raises(ValueError, match="父节点不存在"):
        dag.add_node("A", parents=["nope"])


def test_cycle_edge_raises():
    dag = TaskDAG("root?")
    a = dag.add_node("A")
    b = dag.add_node("B", parents=[a])
    with pytest.raises(ValueError, match="成环"):
        dag.add_edge(b, a)


def test_mermaid_and_dict():
    dag = TaskDAG("root?")
    a = dag.add_node("A")
    dag.add_node("B", parents=[a])
    assert dag.to_mermaid().startswith("graph TD")
    assert "-->" in dag.to_mermaid()
    assert len(dag.to_dict()["nodes"]) == 2