"""MindSearch 的任务 DAG（子问题图）——纯标准库，零重依赖。

与 LangGraph 执行图是两回事：执行图静态（START→planner→END），这张图在 planner
节点内部逐轮增长，最终落盘为 dag/<qid>.json 供报告可视化。所有非法操作显式 raise。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DagNode:
    qid: str
    query: str
    parents: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    status: str = "pending"          # pending | running | done | failed
    answer: str = ""
    sources: list[dict] = field(default_factory=list)


class TaskDAG:
    def __init__(self, root_query: str) -> None:
        self.root_query = root_query
        self.nodes: dict[str, DagNode] = {}
        self._seq = 0

    def _next_id(self) -> str:
        qid = f"n{self._seq}"
        self._seq += 1
        return qid

    def add_node(self, query: str, parents: list[str] | None = None,
                 qid: str | None = None) -> str:
        qid = qid or self._next_id()
        if qid in self.nodes:
            raise ValueError(f"TaskDAG 节点重复: {qid!r}")
        parents = list(parents or [])
        unknown = [p for p in parents if p not in self.nodes]
        if unknown:
            raise ValueError(f"TaskDAG 父节点不存在: {unknown}（节点 {qid!r}）")
        self.nodes[qid] = DagNode(qid=qid, query=query, parents=parents)
        for p in parents:
            self.nodes[p].children.append(qid)
        return qid

    def add_edge(self, src: str, dst: str) -> None:
        for q in (src, dst):
            if q not in self.nodes:
                raise ValueError(f"TaskDAG 节点不存在: {q!r}")
        if src in self.nodes[dst].parents:
            return
        if src == dst or self._reachable(dst, src):
            raise ValueError(f"TaskDAG 加边会成环: {src!r} -> {dst!r}")
        self.nodes[dst].parents.append(src)
        self.nodes[src].children.append(dst)

    def _reachable(self, start: str, target: str) -> bool:
        stack, seen = [start], set()
        while stack:
            cur = stack.pop()
            if cur == target:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(self.nodes[cur].children)
        return False

    def ready(self) -> list[str]:
        """父节点全完成、自己还没开始的节点（可并发执行）。"""
        return [q for q, n in self.nodes.items()
                if n.status == "pending"
                and all(self.nodes[p].status == "done" for p in n.parents)]

    def pending(self) -> list[str]:
        return [q for q, n in self.nodes.items() if n.status in ("pending", "running")]

    def mark_running(self, qid: str) -> None:
        self._require(qid).status = "running"

    def mark_done(self, qid: str, answer: str, sources: list[dict] | None = None) -> None:
        node = self._require(qid)
        node.status, node.answer = "done", answer
        node.sources = list(sources or [])

    def mark_failed(self, qid: str, reason: str = "") -> None:
        node = self._require(qid)
        node.status, node.answer = "failed", reason

    def _require(self, qid: str) -> DagNode:
        if qid not in self.nodes:
            raise ValueError(f"TaskDAG 节点不存在: {qid!r}")
        return self.nodes[qid]

    def topological(self) -> list[str]:
        indeg = {q: len(n.parents) for q, n in self.nodes.items()}
        queue = [q for q, d in indeg.items() if d == 0]
        order: list[str] = []
        while queue:
            cur = queue.pop(0)
            order.append(cur)
            for child in self.nodes[cur].children:
                indeg[child] -= 1
                if indeg[child] == 0:
                    queue.append(child)
        if len(order) != len(self.nodes):
            raise ValueError("TaskDAG 成环，无法拓扑排序")
        return order

    def to_dict(self) -> dict:
        return {"root_query": self.root_query,
                "nodes": [{"qid": n.qid, "query": n.query, "parents": n.parents,
                           "status": n.status, "answer": n.answer, "sources": n.sources}
                          for n in (self.nodes[q] for q in self.topological())]}

    @classmethod
    def from_dict(cls, data: dict) -> "TaskDAG":
        """从 to_dict() 的产物还原（节点需按拓扑序，先父后子）。"""
        if not isinstance(data, dict):
            raise TypeError(f"TaskDAG.from_dict 需要 dict，得到 {type(data).__name__}")
        nodes = data.get("nodes") or []
        if not isinstance(nodes, list):
            raise ValueError("from_dict: nodes 必须是数组")

        dag = cls(str(data.get("root_query", "")))

        # 1. 先建节点：to_dict 已按拓扑序输出，保证父节点先存在
        for item in nodes:
            qid = item.get("qid")
            if not isinstance(qid, str) or not qid:
                raise ValueError(f"from_dict: 非法 qid {qid!r}")
            dag.add_node(str(item.get("query", "")),
                        parents=list(item.get("parents") or []), qid=qid)

        # 2. 再恢复状态（add_node 会把 status 一律重置成 pending）
        for item in nodes:
            qid, status = item["qid"], item.get("status", "pending")
            if status == "done":
                dag.mark_done(qid, item.get("answer", ""), list(item.get("sources") or []))
            elif status == "failed":
                dag.mark_failed(qid, item.get("answer", ""))
            elif status not in ("pending", "running"):
                raise ValueError(f"from_dict: 未知 status {status!r}（节点 {qid}）")

        # 3. 修 id 计数器：否则下一条 add_node 会重新生成 n0 → 与已有节点撞车
        used = [int(q[1:]) for q in dag.nodes if q.startswith("n") and q[1:].isdigit()]
        dag._seq = (max(used) + 1) if used else 0
        return dag

    def to_mermaid(self) -> str:
        lines = ["graph TD"]
        for qid, node in self.nodes.items():
            label = node.query.replace('"', "'")[:40]
            lines.append(f'    {qid}["{label}"]')
        for qid, node in self.nodes.items():
            for parent in node.parents:
                lines.append(f"    {parent} --> {qid}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.nodes)
