# MindSearch 接口冻结记录（IFACE v1.1）

> 适用范围：`src/lychee_mas/methods/build/mindsearch/` 与 `build_mindsearch` 系列分支。
> 目的：把"两人怎么对接"固化成代码契约，使 planner 与 searcher 可以真并行。
> 状态：**已冻结**。改动按本文 §5 流程执行，改签名必须双方确认。

---

## 1. 设计定位（一句话）

把 MindSearch 的协作结构映射到本框架的 **build 接缝**：LangGraph 侧只有静态骨架
`START → planner → END`；论文里那张**逐轮增长的任务 DAG** 放在 planner 节点内部，
落盘为 `state["dag"]`。

这样既不违反 `graphview.extract_view` 的「静态 DAG + 唯一终端」约束（prerun / MASPO /
剪枝照常可用），又保留论文「规划即建图」的语义。

### 为什么不把循环画进 LangGraph

`extract_view` 显式拒绝**成环**与**条件分支**（`prerun 只支持 DAG`）。图上有环 →
`optimize_langgraph` 全部失效（MASPO / agentprune / agentdropout 都用不了）。
而且论文的子问题是运行时生成的，静态图无法表达。**这是取舍，不是缺失。**

---

## 2. 文件与所有权

| 文件                  | 导出符号                                                                                        | Owner               | 冻结级别                                    |
| --------------------- | ----------------------------------------------------------------------------------------------- | ------------------- | ------------------------------------------- |
| `dag.py`              | `TaskDAG`、`DagNode`                                                                            | A                   | 签名冻结                                    |
| `interpreter.py`      | `Ops`、`NewNode`、`parse_ops`、`apply_ops`、`plan_with_retry`、`OpsParseError`、`OpsApplyError` | A                   | 签名 + 语义冻结                             |
| `searcher.py`         | `SourceRef`、`SearcherResult`、`WebSearcher`、`LLM/SearchFn/FetchFn`                            | 实现 B / 消费 A     | 签名 + 三条语义冻结                         |
| `state.py`            | `MindSearchState`                                                                               | 共享                | 键名冻结                                    |
| `builder.py`          | `MindSearchGraphBuilder`（注册 `graph_builder/mindsearch`）                                     | A                   | 签名冻结                                    |
| `prompts.py`          | `PLANNER_PROMPT`                                                                                | A                   | 不冻结（必须保留 `{question}`/`{context}`） |
| `searcher_prompts.py` | `REWRITE_PROMPT`/`SELECT_PROMPT`/`SUMMARIZE_PROMPT`                                             | B                   | 不冻结                                      |
| `__init__.py`         | 上述符号再导出                                                                                  | A（新增导入需通知） | —                                           |

**为什么 `prompts.py` 只有 planner**：`builder` 把它注入 `AgentSpec.system_prompt`，
prerun（MASPO apply）会原地改写它，所以它属于图契约；searcher 的提示不进契约，
归 B 自己的 `searcher_prompts.py`。

---

## 3. 公开契约

### 3.1 任务 DAG（`dag.py`）

```python
dag = TaskDAG(root_query)
qid = dag.add_node(query, parents=[...])      # -> "n0", "n1", ...
dag.add_edge(src, dst)                        # 幂等；成环 -> ValueError
dag.ready()                                   # 父节点全 done 且自身 pending
dag.mark_running(qid) / mark_done(qid, answer, sources) / mark_failed(qid, reason)
dag.topological()                             # 成环 -> ValueError
dag.to_dict() / dag.to_mermaid()              # 落盘 / 报告可视化
len(dag)
```
