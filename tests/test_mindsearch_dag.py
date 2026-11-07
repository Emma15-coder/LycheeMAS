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
    with pytest.raises(ValueError, match="unknown parent"):
        dag.add_node("A", parents=["nope"])


def test_cycle_edge_raises():
    dag = TaskDAG("root?")
    a = dag.add_node("A")
    b = dag.add_node("B", parents=[a])
    with pytest.raises(ValueError, match="cycle"):
        dag.add_edge(b, a)


def test_mermaid_and_dict():
    dag = TaskDAG("root?")
    a = dag.add_node("A")
    dag.add_node("B", parents=[a])
    assert dag.to_mermaid().startswith("graph TD")
    assert "-->" in dag.to_mermaid()
    assert len(dag.to_dict()["nodes"]) == 2

def test_dag_roundtrip():
        dag = TaskDAG("root?")
        a = dag.add_node("A")
        dag.mark_done(a, "ans-a", [{"url": "https://x"}])
        b = dag.add_node("B", parents=[a])

        back = TaskDAG.from_dict(dag.to_dict())
        assert back.to_dict() == dag.to_dict()   # 结构 + 状态 + 来源全部一致
        assert back.ready() == [b]               # 调度状态也恢复了
        assert back.add_node("C") == "n2"        # 新 id 不与已恢复的冲突

def test_from_dict_unknown_status_raises():
    blob = {"root_query": "r", "nodes": [
        {"qid": "n0", "query": "A", "parents": [], "status": "zombie", "answer": "", "sources": []}]}
    with pytest.raises(ValueError, match="unknown status"):
        TaskDAG.from_dict(blob)