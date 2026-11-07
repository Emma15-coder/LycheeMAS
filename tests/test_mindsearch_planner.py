"""planner 的离线测试：假 searcher + 脚本化 LLM，全部同步驱动（asyncio.run）。"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from lychee_mas.core.types import AgentSpec
from lychee_mas.methods.build.mindsearch.dag import TaskDAG
from lychee_mas.methods.build.mindsearch.interpreter import apply_ops, parse_ops
from lychee_mas.methods.build.mindsearch.planner import (
    _summarize, build_context, build_plan_prompt, make_planner_node)
from lychee_mas.methods.build.mindsearch.searcher import SearcherResult, SourceRef

SPEC = AgentSpec(name="planner", role="planner", system_prompt="P {question} {context}")

FINISH = '{"finish": "Answer: 42\\n因为 6x7=42。来源：https://x"}'
GIVEUP = '{"finish": "Answer: 未能确认\\n未能确认：播出日期"}'
BAD_CITATION = '{"finish": "Answer: 42\\n来源：https://evil.example/fake"}'


# ---------------------------------------------------------------- 测试替身

def ok(q: str, answer: str) -> SearcherResult:
    return SearcherResult(sub_question=q, status="ok", answer=answer,
                          sources=[SourceRef(url="https://x", title="t")],
                          prompt_tokens=10, completion_tokens=5)


def empty(q: str) -> SearcherResult:
    return SearcherResult(sub_question=q, status="empty")


def err(q: str, msg: str = "HTTP 403") -> SearcherResult:
    return SearcherResult(sub_question=q, status="error", error=msg)


class FakeSearcher:
    """脚本化检索者；越界不抛异常（会被 _retrieve 兜底吞掉），改用 overrun 标志。"""

    def __init__(self, results):
        self.results = list(results)
        self.calls: list[tuple[str, str]] = []
        self.events: list[tuple[str, str]] = []
        self.overrun = False

    async def run(self, sub_question: str, context: str = "") -> SearcherResult:
        self.events.append(("start", sub_question))
        await asyncio.sleep(0.01)                 # 让并发可被观测
        self.events.append(("end", sub_question))
        self.calls.append((sub_question, context))
        if not self.results:
            self.overrun = True
            return err(sub_question, "FAKE_OVER_RUN")
        r = self.results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class ScriptedLLM:
    def __init__(self, texts):
        self.texts = list(texts)
        self.prompts: list[str] = []

    async def __call__(self, prompt: str):
        self.prompts.append(prompt)
        return SimpleNamespace(text=self.texts.pop(0), n_prompt_pos=100, n_gen_tokens=20)


def run(node, state):
    return asyncio.run(node(state))


def make_node(searcher, llm, **kw):
    return make_planner_node(SPEC, True, searcher=searcher, llm=llm, **kw)


def nodes_json(*queries, parents=None) -> str:
    return json.dumps({"nodes": [{"query": q, "parents": list(parents or [])}
                                 for q in queries]})


def assert_no_overrun(searcher: FakeSearcher) -> None:
    assert not searcher.overrun, (
        f"FakeSearcher 被调用 {len(searcher.calls)} 次，超出脚本；"
        f"calls={[c[0] for c in searcher.calls]}")


# ---------------------------------------------------------------- 纯函数

def test_context_parent_first_then_root():
    dag = TaskDAG("根问题")
    a = dag.add_node("A"); dag.mark_done(a, "结论A")
    b = dag.add_node("B", parents=[a])
    ctx = build_context(dag, b)
    assert ctx.startswith("[n0] 结论A")
    assert ctx.endswith("Root question: 根问题")
    assert "\n---\n" in ctx


def test_context_without_parent_and_unknown_qid():
    dag = TaskDAG("根问题"); a = dag.add_node("A")
    assert "no parent" in build_context(dag, a)
    with pytest.raises(ValueError, match="unknown node"):
        build_context(dag, "n9")


def test_summary_marks_repeated_failures_and_clips():
    dag = TaskDAG("q")
    a = dag.add_node("同一问法"); dag.mark_failed(a, "HTTP 403")
    b = dag.add_node("同一问法"); dag.mark_failed(b, "HTTP 403")
    c = dag.add_node("长答案"); dag.mark_done(c, "结" * 500)
    out = _summarize(dag, limit=50)
    assert "[failed×2]" in out
    assert "…" in out


def test_summary_rows_linked_to_max_nodes():
    """摘要行数与 max_nodes 联动：超过 max_rows 时出现省略行。"""
    dag = TaskDAG("q")
    for i in range(5):
        dag.add_node(f"子问题{i}")
    out = _summarize(dag, max_rows=3)
    assert "2 more nodes not shown" in out


def test_prompt_three_modes():
    dag = TaskDAG("q"); dag.add_node("A")
    assert "At most 3 sub-questions" in build_plan_prompt("q", dag)
    assert "turn/budget limit is reached" in build_plan_prompt("q", dag, finish_only=True)
    assert "cannot confirm" in build_plan_prompt("q", dag, give_up=True)
    assert "previous output was invalid" in build_plan_prompt("q", dag, feedback="unknown field foo")


# ---------------------------------------------------------------- 主循环

def test_two_rounds_converge():
    searcher = FakeSearcher([ok("A", "a1"), ok("B", "b2")])
    llm = ScriptedLLM([nodes_json("A", "B"), FINISH])
    state = run(make_node(searcher, llm), {"question": "Q"})

    assert state["stop_reason"] == "finish"
    assert state["truncated"] is False
    assert state["answer"] == "42"
    assert state["format_ok"] is True
    assert state["turns"] == 2
    assert state["prompt_tokens"] > 0 and state["completion_tokens"] > 0
    assert [c[0] for c in searcher.calls] == ["A", "B"]
    assert_no_overrun(searcher)

    dag = TaskDAG.from_dict(state["dag"])
    assert len(dag) == 3                                  # n0 + n1 + response
    final = [q for q in dag.nodes if dag.nodes[q].query == "[FINAL RESPONSE]"]
    assert len(final) == 1
    assert all(dag.nodes[p].status == "done" for p in dag.nodes[final[0]].parents)


def test_retrieve_is_parallel():
    searcher = FakeSearcher([ok("A", "a"), ok("B", "b")])
    llm = ScriptedLLM([nodes_json("A", "B"), FINISH])
    run(make_node(searcher, llm), {"question": "Q"})
    # 并发时两个 start 相邻；串行会是 start,end,start,end
    assert [e[0] for e in searcher.events[:2]] == ["start", "start"]
    assert_no_overrun(searcher)


def test_plan_error_after_retries():
    searcher = FakeSearcher([])
    llm = ScriptedLLM(["not json", "still bad", "bad again"])
    state = run(make_node(searcher, llm, max_retry=2), {"question": "Q"})

    assert state["stop_reason"] == "plan_error"
    assert state["truncated"] is True
    assert "[UNRESOLVED]" in state["outputs"]["planner"]
    assert searcher.calls == []                           # 规划都没成功，不应检索


def test_max_turns_forces_finish():
    searcher = FakeSearcher([])
    llm = ScriptedLLM([nodes_json("A"), FINISH])          # 第 2 段是强制收尾
    state = run(make_node(searcher, llm, max_turns=1), {"question": "Q"})

    assert state["stop_reason"] == "max_turns"
    assert state["truncated"] is True
    assert state["answer"] == "42"
    assert searcher.calls == []                           # 只有 1 轮，还没检索就收尾


def test_per_turn_truncation():
    # 第 2 轮会并发检索 3 个已建节点 → 必须准备 3 条结果，否则 overrun
    searcher = FakeSearcher([ok("A", "a"), ok("B", "b"), ok("C", "c")])
    llm = ScriptedLLM([nodes_json("A", "B", "C", "D"), FINISH])
    state = run(make_node(searcher, llm, max_subq_per_turn=3), {"question": "Q"})

    dag = TaskDAG.from_dict(state["dag"])
    created = [q for q in dag.nodes if dag.nodes[q].query in {"A", "B", "C", "D"}]
    assert len(created) == 3                              # 第 4 个被上限截掉
    assert state["truncated"] is True
    assert_no_overrun(searcher)


def test_stalled_breaks_when_new_nodes_are_blocked():
    # 轮1：建 A；轮2：A 检索失败，规划者把 B 挂到 A 下（不可执行）；
    # 轮3：规划者又把 C 挂到 A 下 → ready() 仍为空 → stalled 停止
    plan1 = nodes_json("A")
    plan2 = json.dumps({"nodes": [{"query": "B", "parents": ["n0"]}]})
    plan3 = json.dumps({"nodes": [{"query": "C", "parents": ["n0"]}]})
    searcher = FakeSearcher([err("A")])
    llm = ScriptedLLM([plan1, plan2, plan3, FINISH])      # 第 4 段是强制收尾

    state = run(make_node(searcher, llm, max_turns=5), {"question": "Q"})

    assert state["stop_reason"] == "stalled"
    assert state["truncated"] is True
    assert state["answer"] == "42"
    assert_no_overrun(searcher)


def test_failed_blocks_child_and_replans():
    plan1 = json.dumps({"nodes": [{"query": "A", "parents": []},
                                  {"query": "B", "parents": ["n0"]}]})
    plan2 = json.dumps({"nodes": [{"query": "A2", "parents": []}]})
    searcher = FakeSearcher([err("A"), ok("A2", "找到了")])
    llm = ScriptedLLM([plan1, plan2, FINISH])

    state = run(make_node(searcher, llm), {"question": "Q"})
    dag = TaskDAG.from_dict(state["dag"])

    assert dag.nodes["n0"].status == "failed"
    assert dag.nodes["n1"].status == "pending"            # 子节点被冻结
    assert dag.nodes["n2"].status == "done"               # 替代节点跑通
    assert [c[0] for c in searcher.calls] == ["A", "A2"]
    assert state["stop_reason"] == "finish"
    assert_no_overrun(searcher)


def test_giveup_after_fail_streak():
    searcher = FakeSearcher([empty("A"), empty("B"), err("C"), err("D")])
    llm = ScriptedLLM([nodes_json("A", "B"), nodes_json("C", "D"), GIVEUP])
    state = run(make_node(searcher, llm, max_fail_streak=2), {"question": "Q"})

    assert state["stop_reason"] == "giveup"
    assert "未能确认" in state["outputs"]["planner"]
    assert_no_overrun(searcher)

    dag = TaskDAG.from_dict(state["dag"])
    final = [q for q in dag.nodes if dag.nodes[q].query == "[FINAL RESPONSE]"][0]
    # 响应节点只连 done 节点（n0/n1 空结果仍算 done；n2/n3 failed 不连）
    assert set(dag.nodes[final].parents) == {"n0", "n1"}


def test_resume_resets_running():
    dag = TaskDAG("Q")
    a = dag.add_node("A")
    dag.mark_running(a)                                   # 模拟上次中断
    searcher = FakeSearcher([ok("A", "a1")])
    llm = ScriptedLLM([FINISH])
    state = run(make_node(searcher, llm), {"question": "Q", "dag": dag.to_dict()})

    assert searcher.calls and searcher.calls[0][0] == "A"  # 被重新调度
    assert state["stop_reason"] == "finish"
    assert_no_overrun(searcher)


# ---------------------------------------------------------------- 新增：finish 合法性 + 引用校验

def test_finish_rejected_without_retrieval_then_forced_expand():
    """强模型第 1 轮就想直接回答：拒绝 → 连续拒绝后程序化建节点，保证检索发生。"""
    searcher = FakeSearcher([ok("Q", "a1")])
    llm = ScriptedLLM([FINISH, FINISH, FINISH, FINISH])      # 前三次都试图直接答
    state = run(make_node(searcher, llm, max_turns=6), {"question": "Q"})

    assert state["stop_reason"] == "finish"
    assert searcher.calls, "兜底建节点后必须发生一次检索"
    assert state["citation_ok"] is True
    dag = TaskDAG.from_dict(state["dag"])
    assert [q for q in dag.nodes if dag.nodes[q].query == "Q"], "根问题应被兜底建成节点"
    assert_no_overrun(searcher)


def test_citation_gate_rejects_unknown_url_and_reanswers():
    """答案引用了未检索到的 URL：拒绝并回灌，模型改用检索来源后通过。"""
    searcher = FakeSearcher([ok("A", "a1")])
    llm = ScriptedLLM([nodes_json("A"), BAD_CITATION, FINISH])
    state = run(make_node(searcher, llm), {"question": "Q"})

    assert state["stop_reason"] == "finish"
    assert state["citation_ok"] is True
    assert "evil.example" not in state["outputs"]["planner"]
    assert_no_overrun(searcher)


def test_citation_failure_recorded_when_retry_exhausted():
    """引用不合规且没有重答预算：仍收尾，但显式记录 citation_ok=False。"""
    searcher = FakeSearcher([ok("A", "a1")])
    llm = ScriptedLLM([nodes_json("A"), BAD_CITATION])
    state = run(make_node(searcher, llm, max_citation_retries=0), {"question": "Q"})

    assert state["stop_reason"] == "finish"
    assert state["citation_ok"] is False
    assert any("evil.example" in u for u in state["citation_unknown"])


def test_duplicate_subquestion_is_skipped():
    """归一化后与已有节点相同的子问题不入图，并记 warnings。"""
    dag = TaskDAG("Q")
    dag.add_node("A")
    ops = parse_ops(nodes_json("a ", "B"))
    created, _ = apply_ops(dag, ops)
    assert len(created) == 1
    assert any("duplicate" in w for w in ops.warnings)


def test_only_duplicates_converge_to_redundant():
    """连续只提重复子问题：不推进 DAG，达到阈值后以 redundant 收尾。"""
    searcher = FakeSearcher([ok("A", "a1")])
    llm = ScriptedLLM([nodes_json("A"), nodes_json("A"), nodes_json("A"), FINISH])
    state = run(make_node(searcher, llm, max_redundant=2), {"question": "Q"})

    assert state["stop_reason"] == "redundant"
    assert "[UNRESOLVED]" in state["outputs"]["planner"]
    assert len(state["dag"]["nodes"]) <= 2          # 一个子问题 + FINAL
    assert_no_overrun(searcher)


def test_system_prompt_is_rendered_into_plan_prompt():
    """spec.system_prompt 必须真正进入每轮提示（MASPO 改写才有效）。"""
    dag = TaskDAG("Q")
    prompt = build_plan_prompt("Q", dag, system_prompt="ROLE {question}/{context}")
    assert "ROLE Q" in prompt
    assert build_plan_prompt("Q", dag, system_prompt="")           # 空提示不报错


def test_summary_lists_source_urls():
    """来源 URL 必须进入提示，否则规划者无法引用检索结果。"""
    dag = TaskDAG("q")
    a = dag.add_node("A")
    dag.mark_done(a, "ans", [{"url": "https://x", "title": "t"}])
    assert "https://x" in _summarize(dag)


def test_final_node_collects_sources():
    """响应节点汇总已完成节点的来源（去重）。"""
    searcher = FakeSearcher([ok("A", "a1")])
    llm = ScriptedLLM([nodes_json("A"), FINISH])
    state = run(make_node(searcher, llm), {"question": "Q"})
    dag = TaskDAG.from_dict(state["dag"])
    final = [q for q in dag.nodes if dag.nodes[q].query == "[FINAL RESPONSE]"][0]
    assert dag.nodes[final].sources
    assert dag.nodes[final].sources[0]["url"] == "https://x"


FINISH_DEPS = '{"finish": "Answer: 42\\n来源：https://x", "response_parents": ["n0"]}'


def test_response_node_uses_model_parents():
    """论文语义：response 节点的依赖由模型指定，而不是自动连全部节点。"""
    searcher = FakeSearcher([ok("A", "a1"), ok("B", "b1")])
    llm = ScriptedLLM([nodes_json("A", "B"), FINISH_DEPS])
    state = run(make_node(searcher, llm), {"question": "Q"})

    dag = TaskDAG.from_dict(state["dag"])
    resp = [q for q in dag.nodes if dag.nodes[q].query == "[FINAL RESPONSE]"][0]
    assert dag.nodes[resp].parents == ["n0"]


def test_invalid_response_parents_rejected_then_auto_connected():
    """响应依赖非法：先回灌一次；仍非法则退回"连接全部已完成节点"。"""
    searcher = FakeSearcher([ok("A", "a1")])
    bad = '{"finish": "Answer: 42\\n来源：https://x", "response_parents": ["n9"]}'
    llm = ScriptedLLM([nodes_json("A"), bad, FINISH])
    state = run(make_node(searcher, llm), {"question": "Q"})

    assert state["stop_reason"] == "finish"
    dag = TaskDAG.from_dict(state["dag"])
    resp = [q for q in dag.nodes if dag.nodes[q].query == "[FINAL RESPONSE]"][0]
    assert dag.nodes[resp].parents == ["n0"]


def test_blocked_child_visible_in_summary():
    """父节点失败后，子节点永远不 ready：摘要必须标 blocked，规划者才知道要绕开。"""
    dag = TaskDAG("Q")
    a = dag.add_node("A")
    dag.mark_failed(a, "HTTP 403")
    dag.add_node("B", parents=[a])
    assert "blocked by n0" in _summarize(dag)


def test_context_respects_total_budget():
    """多父节点时按总预算均分，避免 prompt 爆炸。"""
    dag = TaskDAG("Q")
    ids = [dag.add_node(f"P{i}") for i in range(6)]
    for q in ids:
        dag.mark_done(q, "x" * 2000)
    c = dag.add_node("C", parents=ids)
    assert len(build_context(dag, c)) <= 2400 + 200
