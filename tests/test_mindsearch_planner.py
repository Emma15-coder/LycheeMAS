"""planner 的离线测试：假 searcher + 脚本化 LLM，全部同步驱动（asyncio.run）。"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from lychee_mas.core.types import AgentSpec
from lychee_mas.methods.build.mindsearch.dag import TaskDAG
from lychee_mas.methods.build.mindsearch.planner import (
    _summarize, build_context, build_plan_prompt, make_planner_node)
from lychee_mas.methods.build.mindsearch.searcher import SearcherResult, SourceRef

SPEC = AgentSpec(name="planner", role="planner", system_prompt="P {question} {context}")

FINISH = '{"finish": "Answer: 42\\n因为 6x7=42"}'
GIVEUP = '{"finish": "Answer: 未能确认\\n未能确认：播出日期"}'


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
    assert ctx.endswith("根问题：根问题")
    assert "\n---\n" in ctx


def test_context_without_parent_and_unknown_qid():
    dag = TaskDAG("根问题"); a = dag.add_node("A")
    assert "无父任务" in build_context(dag, a)
    with pytest.raises(ValueError, match="节点不存在"):
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
    assert "另有 2 个节点未展示" in out


def test_prompt_three_modes():
    dag = TaskDAG("q"); dag.add_node("A")
    assert "最多 3 个子问题" in build_plan_prompt("q", dag)
    assert "已到轮数" in build_plan_prompt("q", dag, finish_only=True)
    assert "未能确认" in build_plan_prompt("q", dag, give_up=True)
    assert "上一次输出不合法" in build_plan_prompt("q", dag, feedback="未知字段 foo")


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
    assert len(dag) == 3                                  # n0 + n1 + [FINAL RESPONSE]
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
    assert "[未收敛]" in state["outputs"]["planner"]
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