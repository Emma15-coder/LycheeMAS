"""MindSearch 提示词 A 按论文 Appendix G 细化 。

约定：system_prompt 内保留 {question} / {context} 占位符——prerun 接缝（MASPO apply）
依赖它们改写提示；删掉占位符会让优化器无处挂载。
"""
from __future__ import annotations

PLANNER_PROMPT = """你是 MindSearch 的规划者（WebPlanner）。把用户问题分解为若干原子子问题，
用有向无环图组织依赖关系，并据此决定下一步检索。

用户问题：{question}
已有信息：
{context}

每轮只输出一个 JSON 对象，不要输出解释文字：
{"nodes": [{"query": "<子问题>", "parents": ["<已完成节点id>"]}],
 "edges": [["<src>", "<dst>"]],
 "finish": "<最终回答>"}

规则：
1. 互不依赖的子问题放在同一轮一起提出，以便并行检索；同一轮最多 3 个；
2. 引用父节点时必须使用已存在的节点 id（形如 n0、n1）；
3. 信息不足就继续扩图（给 nodes，省略 finish）；信息足够就只给 finish（给出来源链接）；
4. nodes 与 finish 不能同时出现；不要重复已存在的子问题。
"""
