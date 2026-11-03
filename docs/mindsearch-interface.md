# MindSearch 初始开发接口（IFACE v1.3 · 完整版）

> **地位**：本文件是 build 接缝接口的唯一事实来源，取代此前所有草稿。
> **适用范围**：`src/lychee_mas/methods/build/mindsearch/` 与 `build_mindsearch*` 分支。
> **不包含**：数据集与打分口径（见 `docs/mindsearch-eval-protocol.md`）、searcher / planner 的内部实现。
> **状态**：已冻结。改动按 §5 流程执行；改签名必须双方确认。

---

## 1. 设计定位与三条关键取舍

把 MindSearch 的协作结构映射到本框架的 **build 接缝**：LangGraph 侧只有静态骨架
`START → planner → END`；论文里那张**逐轮增长的任务 DAG** 放在 planner 节点内部，
落盘为 `state["dag"]`。

| 取舍                     | 做法                                                            | 原因                                                                                                                                                    |
| ------------------------ | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 不在图里画循环           | 静态壳 + 节点内 DAG                                             | `graphview.extract_view` 只接受**静态 DAG + 唯一终端**；图上有环或条件分支，`optimize_langgraph` 全部失效（MASPO / agentprune / agentdropout 都用不了） |
| 规划动作不用 Python 代码 | 每轮一个 JSON ops + 严格校验 + 错误回灌重试                     | 不引入 `exec` 的安全面；错误可精确定位并反馈（等价论文附录 E 的"用执行反馈纠错"）                                                                       |
| 短答案由生产者产出       | `answer_format.format_answer()` 把 `finish` 拆成短答案 + 长答案 | 打分逻辑（`eval/metrics.py`）一行不动；eval 只拿成品                                                                                                    |

**这是取舍，不是缺失**：论文的子问题是运行时生成的，静态图无法表达；而保住 prerun 能力
（可被 MASPO 优化、可被剪枝）比"图上看起来像论文"更有价值。

---

## 2. 文件、所有权与落地状态

### 2.1 文件与所有权

| 文件                  | 导出符号                                                                                        | Owner               | 冻结级别                                             |
| --------------------- | ----------------------------------------------------------------------------------------------- | ------------------- | ---------------------------------------------------- |
| `dag.py`              | `TaskDAG`、`DagNode`                                                                            | A                   | 签名冻结                                             |
| `interpreter.py`      | `Ops`、`NewNode`、`parse_ops`、`apply_ops`、`plan_with_retry`、`OpsParseError`、`OpsApplyError` | A                   | 签名 + 语义冻结                                      |
| `searcher.py`         | `SourceRef`、`SearcherResult`、`WebSearcher`、`LLM/SearchFn/FetchFn`                            | 实现 B / 消费 A     | 签名 + 三条语义冻结                                  |
| `state.py`            | `MindSearchState`                                                                               | 共享                | **键名冻结**                                         |
| `builder.py`          | `MindSearchGraphBuilder`（注册 `graph_builder/mindsearch`）                                     | A                   | 签名冻结                                             |
| `answer_format.py`    | `FormattedAnswer`、`format_answer`                                                              | A                   | 签名冻结                                             |
| `prompts.py`          | `PLANNER_PROMPT`                                                                                | A                   | 不冻结（必须保留 `{question}` / `{context}` 占位符） |
| `planner.py`          | `make_planner_node`、`planner_node_factory`                                                     | A                   | 签名冻结（实现待落地）                               |
| `searcher_prompts.py` | `REWRITE_PROMPT` / `SELECT_PROMPT` / `SUMMARIZE_PROMPT`                                         | **仅 B**            | 不冻结                                               |
| `__init__.py`         | 上述符号再导出                                                                                  | A（新增导入需通知） | —                                                    |

**为什么 `prompts.py` 只有 planner**：`builder` 把它注入 `AgentSpec.system_prompt`，
prerun（MASPO apply）会**原地改写**它，所以它属于图契约；searcher 的提示不进入图契约，
归 B 自己的 `searcher_prompts.py`。

### 2.2 落地状态总览（先看这张表，避免找错文件）

| 项                                                                                      | 状态               | 位置 / 分支                    |
| --------------------------------------------------------------------------------------- | ------------------ | ------------------------------ |
| `TaskDAG` / ops 解析 / `SearcherResult` / `MindSearchState` / builder / `answer_format` | 已落地（首次推送） | 集成分支 `build_mindsearch`    |
| `WebSearcher.run()` 实现                                                                | 未落地             | B：`build_mindsearch_searcher` |
| `searcher_prompts.py`                                                                   | 未落地             | B：同上（**不属于集成分支**）  |
| `planner.py`（循环 + 上限 + 并发）                                                      | 未落地             | A：`build_mindsearch_planner`  |
| `TaskDAG.ready()` 把 `failed` 视为终态                                                  | 提案 v1.3          | A：planner 分支，随实现落地    |
| 3 个 benchmark loader + 注册                                                            | 未落地             | B：`eval_mindsearch`           |
| `scripts/run_mindsearch.py` 等入口                                                      | 未落地             | A/B 各自子分支                 |

---

## 3. 公开契约

### 3.1 任务 DAG（`dag.py`）

```python
dag = TaskDAG(root_query)
qid = dag.add_node(query, parents=[...])       # -> "n0", "n1", ...
dag.add_edge(src, dst)                         # 幂等；成环 -> ValueError
dag.ready()                                    # 父节点终态且自身 pending
dag.pending()
dag.mark_running(qid) / mark_done(qid, answer, sources) / mark_failed(qid, reason)
dag.topological()                              # 成环 -> ValueError
dag.to_dict() / dag.to_mermaid()               # 落盘 dag/<qid>.json / 报告可视化
dag.from_dict()                                # 按拓扑顺序恢复，后置恢复status/answer/sources，重置id计数器，未知status显示ValueError
len(dag)
```

- 节点 id 由 DAG 生成（`n0`、`n1`…），**调用方不得自定义**，避免与 LLM 幻觉出的 id 冲突。
- 当前 `ready()` 只把 `done` 当作父节点终态；**v1.3 提案**：`failed` 也算终态，
  否则一个检索失败会永久堵死其子节点。

### 3.2 规划指令 ops（`interpreter.py`）

LLM 每轮只输出一个 JSON 对象：

```json
{
  "nodes": [{ "query": "子问题", "parents": ["n0"] }],
  "edges": [["n0", "n1"]],
  "finish": "Answer: <短答案>\n<解释与来源>"
}
```

| 规则                                                                    | 失败行为                          |
| ----------------------------------------------------------------------- | --------------------------------- |
| 字段白名单：只允许 `nodes` / `edges` / `finish`                         | `OpsParseError`                   |
| `query` 必须非空字符串；`parents` / `edges` 必须引用**已存在**的节点 id | `OpsParseError` / `OpsApplyError` |
| `nodes` 与 `finish` **互斥**，且不能都为空                              | `OpsParseError`                   |
| 允许前后缀说明与 `fence`，取第一个平衡的 `{}`                           | 解析失败 → 回灌重试               |

```python
ops = parse_ops(llm_text)                                  # -> Ops
created, truncated = apply_ops(dag, ops, max_subq_per_turn=3)
ops, created, truncated = await plan_with_retry(llm, prompt_builder, dag,
                                                max_retry=2, max_subq_per_turn=3)
```

`plan_with_retry` 的行为：解析/应用失败 → 把错误文本回灌提示词重试，最多 `max_retry` 次；
仍失败则**显式抛出 `OpsParseError`**（绝不静默返回空规划）。

### 3.3 检索契约（`searcher.py`）

```python
@dataclass(frozen=True)
class SourceRef:                 # sources 里只允许这三种键
    url: str
    title: str = ""
    snippet: str = ""

@dataclass
class SearcherResult:
    sub_question: str
    status: str = "ok"           # ok | empty | error
    answer: str = ""
    sources: list[SourceRef] = []
    queries: list[str] = []
    pages_read: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""              # status == "error" 时必填

    def is_ok(self) -> bool: ...
```

**三条冻结语义**

1. **失败不抛异常**：网络/抓取失败返回 `status="error"` + `error` 文本，`run()` 内部最多重试 1 次；
   由 planner 决定重规划还是 `mark_failed`。本层不假装成功，也不炸掉整轮。
2. **空结果**：检索到了但没有可用内容 → `status="empty"`、`answer=""`。
3. **context 语义**：固定为 `"父节点 answer" + "\n---\n" + "根问题"`，**不含**兄弟节点与全部历史。

### 3.4 状态键（`state.py`）

| 键                                    | 类型           | 语义                                                 |
| ------------------------------------- | -------------- | ---------------------------------------------------- |
| `question`                            | str            | 根问题                                               |
| `outputs`                             | dict[str, str] | 节点名 → 最终文本；`outputs["planner"]` 即**长答案** |
| `dag`                                 | dict           | `TaskDAG.to_dict()`                                  |
| `turns`                               | int            | 已执行的规划轮数                                     |
| `truncated`                           | bool           | 是否因上限被截断（轮数 / 子问题数 / 节点数）         |
| `prompt_tokens` / `completion_tokens` | int            | 成本轴（planner + searcher 合计）                    |
| `error`                               | str            | 非空表示本次运行失败（显式记录）                     |
| `answer`                              | str            | **短答案**（打分口径）                               |
| `format_ok`                           | bool           | `finish` 是否符合 `Answer:` 规范（prompt 遵从度）    |

### 3.5 构建器（`builder.py`）

```python
sg = build_langgraph(method="mindsearch", node_factory=..., state_schema=None)
```

| 参数                              | 说明                                                                |
| --------------------------------- | ------------------------------------------------------------------- |
| `node_factory(spec, is_terminal)` | 由实验方注入；planner 节点收到 `is_terminal=True`                   |
| `state_schema`                    | 默认使用 `MindSearchState`（可注入同形状 schema）                   |
| `planner_prompt`                  | 默认 `PLANNER_PROMPT`，注入 `spec.system_prompt`（可被 MASPO 改写） |
| `planner_name`                    | 默认 `"planner"`                                                    |

校验：缺 `node_factory` 或 `state_schema` → `ValueError`；`node_factory` 不可调用 → `TypeError`。
`build()` 内部才 `from langgraph.graph import ...`（**必须惰性导入**，否则 `make selfcheck` 失败）。

### 3.6 答案格式化（`answer_format.py`）

```python
@dataclass(frozen=True)
class FormattedAnswer:
    short: str         # 给打分用的短答案
    long: str          # 完整回答（含解释与来源）
    format_ok: bool    # 是否带规范的 Answer: 行

def format_answer(text: str) -> FormattedAnswer: ...
```

规则：

- 优先取第一条 `Answer: ...`（容忍 markdown 加粗、全角冒号、行首列表符号）；
- 找不到则退化为**最后一行非空文本**，并置 `format_ok=False`；
- 空文本返回 `("", "", False)`。

### 3.7 规划节点工厂（`planner.py`，签名冻结、实现待落地）

```python
def planner_node_factory(*, searcher, llm,
                         max_turns: int = 10, max_subq_per_turn: int = 3,
                         max_nodes: int = 40, max_concurrency: int = 8,
                         wall_clock_s: float = 300.0, max_retry: int = 2,
                         ) -> Callable[[AgentSpec, bool], Node]: ...
```

- 返回的 `node_factory(spec, is_terminal)` 供 `build_langgraph` 使用；
- 节点必须 `async def node(state) -> dict`，返回**部分更新**：
  `{outputs, dag, answer, format_ok, turns, truncated, prompt_tokens, completion_tokens}`；
- `is_terminal=False` 时直接 `raise ValueError`（本方法只有 planner 一个节点）。

---

## 4. 最小用法

```python
from lychee_mas.plugins import build_langgraph
from lychee_mas.methods.build.mindsearch import MindSearchState

# ① 离线 / 单测：假节点
def fake_node(spec, is_terminal):
    async def node(state):
        return {"outputs": {**state.get("outputs", {}), spec.name: "Answer: 4"},
                "answer": "4", "format_ok": True}
    return node

sg = build_langgraph(method="mindsearch", node_factory=fake_node,
                     state_schema=MindSearchState)

# ② 真实运行：由脚本注入 planner_node_factory(searcher=..., llm=...)
```

```python
# 脚本层（B 负责）读取短答案的方式
prediction = state["answer"]              # 打分口径
prediction_raw = state["outputs"]["planner"]   # 长答案，落盘留档
format_ok = state["format_ok"]            # 聚合为 format_ok_rate
```

---

## 5. 变更流程（冻结 ≠ 永不改）

| 级别        | 典型改动                            | 处理                                                                         |
| ----------- | ----------------------------------- | ---------------------------------------------------------------------------- |
| L0 实现级   | 改函数体、加私有 helper、改日志     | 自己分支直接改                                                               |
| L1 兼容扩展 | 新增**带默认值**字段/参数、新增方法 | 单独提交 + docstring 同步；群里说一句                                        |
| L2 破坏性   | 改字段名/类型、删参数、改语义       | 双方确认 → `build_mindsearch_iface_vN` 短分支 → 接口 + 调用方 + 测试一次改完 |
| L3 架构级   | DAG 移出节点、图内循环、换 ops 协议 | 先写一页 ADR，确认后再动                                                     |

**执行规则**

1. 谁的文件谁改（见 §2.1）；集成分支只接受 merge，不接受直接提交（pre-commit hook 已拦）。
2. 接口改动**只在 `build_mindsearch` 上做**，两个子分支随后 `git rebase origin/build_mindsearch`。
3. 一次提交只做"接口改动 + 让测试变绿"，不掺功能。
4. 破坏性改动优先"两步走"：先加新的（带默认值）→ 双方切换 → 下个迭代删旧的。
5. rebase 冲突时：**接口签名以集成分支为准，实现保留自己的**。

---

## 6. 验证

```bash
.venv/bin/python -c "import lychee_mas; print('import ok')"
PYTHONPATH=src .venv/bin/python -m pytest -q -k mindsearch
PYTHONPATH=src .venv/bin/python -c "import lychee_mas; from lychee_mas.core.registry import REGISTRY; \
print(REGISTRY.snapshot()['graph_builder'])"      # 期望 ['mindsearch', 'static']
make lint && make test && make selfcheck && make demo
```

---

## 7. 版本记录

| 版本 | 变更                                                                                                                               | 破坏性 |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------- | ------ |
| v1   | 首次冻结：`TaskDAG` / ops 协议 / `SearcherResult` / `MindSearchState` / builder                                                    | —      |
| v1.1 | searcher 提示移出 `prompts.py` → `searcher_prompts.py`（B 独占）                                                                   | 否     |
| v1.2 | `state` 新增 `answer` / `format_ok`；新增 `answer_format.format_answer()`；登记 `planner_node_factory` 签名；补全 §3.2–§3.7、§4–§6 | 否     |
| v1.3 | 新增 `TaskDAG.from_dict()` （检查点/续跑）`ready()` 的failed终态随planner实现 ；补全 §3.1                                          | 否     |

---

## 8. 与论文的偏差（须写入 PR 与报告）

| 项       | 论文                                     | 本实现                         | 原因                                                            |
| -------- | ---------------------------------------- | ------------------------------ | --------------------------------------------------------------- |
| 规划动作 | Python 代码 + 解释器（code as planning） | JSON ops + 严格校验 + 错误回灌 | 无 `exec` 安全面；错误可定位并反馈（论文附录 E 的等价物）       |
| 执行图   | 运行时动态扩展的 DAG                     | 静态壳 + 节点内 DAG            | `extract_view` 只接受静态 DAG / 唯一终端，保住 prerun 能力      |
| 检索器   | WebSearcher（Bing）                      | 可注入的 search / fetch 回调   | 引擎与 SDK 解耦，便于替换与测试                                 |
| 打分     | HotpotQA 用 GPT-4o judge                 | `exact`（+`f1` 辅助）          | 保持离线纯函数、可复算；详见 `docs/mindsearch-eval-protocol.md` |

---

## 附录 A · 首次推送清单（17 个文件）

```
修改(2)：src/lychee_mas/methods/build/__init__.py、docs/DESIGN.md
新增(8)：methods/build/mindsearch/{__init__,builder,dag,interpreter,prompts,searcher,state,answer_format}.py
新增(1)：configs/build/mindsearch.yaml
新增(4)：tests/test_mindsearch_{dag,interpreter,builder,answer}.py
新增(2)：docs/mindsearch-interface.md、docs/mindsearch-eval-protocol.md 后来又修改为当前版本
```

推送前三条 diff 必须为空：

```bash
git diff upstream/LycheeMASv0.3...HEAD -- src/lychee_mas/plugins src/lychee_mas/eval scripts
```

另外自查是否误带 B 的文件：

```bash
git diff --name-only upstream/LycheeMASv0.3...HEAD | grep -E "searcher_prompts|planner.py"
# 期望：无输出
```

---

## 附录 B · 每日自检命令

```bash
git status -sb                          # 当前分支 + 与远端关系
git diff --cached --stat                # 提交前数文件（17 个是预期）
.venv/bin/python -c "import lychee_mas; print('import ok')"   # 3 秒冒烟
make lint && make test && make selfcheck                      # 三件套
```

---

## 附录 C · 已发生的误操作与防复发

| 事件                                                             | 根因                    | 防法                                                          |
| ---------------------------------------------------------------- | ----------------------- | ------------------------------------------------------------- |
| `searcher_prompts.py` 被提交到集成分支                           | `git add <目录>` 一把梭 | 只用精确路径；提交前 `git diff --cached --stat` 数文件        |
| `test_mind_search_answer.py` 与 `test_mindsearch_answer.py` 并存 | 手误新建                | 命名统一 `test_mindsearch_*`；pytest 会同时收集两个           |
| 文档只存到 §3.1（截断）                                          | 从对话复制长文          | 落盘后 `grep -c '^## '` 自检节数                              |
| 顶层 `import langgraph` 风险                                     | 图省事写在模块头        | 只在函数内 import；`make selfcheck` 必须 `HEAVY LOADED: NONE` |
