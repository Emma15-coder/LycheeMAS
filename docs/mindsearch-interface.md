# MindSearch 初始开发接口（IFACE v1.8 · 完整版）

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
| `planner.py`          | `make_planner_node`、`planner_node_factory`                                                     | A                   | 签名 + 语义冻结（已落地）                            |
| `searcher_prompts.py` | `REWRITE_PROMPT` / `SELECT_PROMPT` / `SUMMARIZE_PROMPT`                                         | **仅 B**            | 不冻结                                               |
| `__init__.py`         | 上述符号再导出                                                                                  | A（新增导入需通知） | —                                                    |
| `scripts/run_mindsearch.py` | 三臂入口（argparse / 落盘 / 打分） | A | 不冻结（实验脚本） |
| `backends/tavily_search.py` | `search` / `fetch` 原语（Tavily Search / Extract） | **仅 B** | 不冻结（实现细节） |

**为什么 `prompts.py` 只有 planner**：`builder` 把它注入 `AgentSpec.system_prompt`，
prerun（MASPO apply）会**原地改写**它，所以它属于图契约；searcher 的提示不进入图契约，
归 B 自己的 `searcher_prompts.py`。

### 2.2 落地状态总览（先看这张表，避免找错文件）

| 项                                                                                      | 状态               | 位置 / 分支                    |
| --------------------------------------------------------------------------------------- | ------------------ | ------------------------------ |
| `TaskDAG` / ops / `SearcherResult` / `MindSearchState` / builder / `answer_format`      | 已落地             | 集成分支 `build_mindsearch`    |
| `planner.py`（循环 + 上限 + 并发 + finish 合法性 + 引用校验 + 去重）                     | **已落地**         | A：`build_mindsearch_planner`  |
| `citation.py`（引用级校验）                                                              | **已落地**         | 同上                           |
| `scripts/run_mindsearch.py`（三臂 + 落盘 + 指标）                                        | **已落地**         | A：同上                        |
| `WebSearcher.run()` 四步实现                                                            | 已落地             | B：`build_mindsearch_searcher` |
| `make_web_searcher` 工厂 + `backends/tavily_backend.py`                                 | 已落地             | B：同上                        |
| `searcher_prompts.py`                                                                   | 已落地             | B：同上                        |
| 3 个 benchmark loader + 注册                                                            | 未落地             | B：`eval_mindsearch`           |
| `analyze_mindsearch_run.py` / `render_dag.py`                                           | 未落地             | B：同上                        |

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
len(dag)
TaskDAG.from_dict(data)                        # 反序列化：检查点 / 续跑
```

- 节点 id 由 DAG 生成（`n0`、`n1`…），**调用方不得自定义**，避免与 LLM 幻觉出的 id 冲突。
- 当前 `ready()` 只把 `done` 当作父节点终态；**v1.3 提案**：`failed` 也算终态，
  否则一个检索失败会永久堵死其子节点。
- `from_dict()` 的语义（v1.3）：节点按拓扑序恢复 → **后置**恢复 `status/answer/sources`
  （`add_node` 会把 status 重置为 `pending`）→ 重置内部 id 计数器（避免新节点与 `n0..nK` 冲突）；
  未知 `status` 显式抛 `ValueError`。

### 3.2 规划指令 ops（`interpreter.py`）

LLM 每轮只输出一个 JSON 对象：

```json
// 形态 A：扩图
{ "nodes": [{ "query": "子问题", "parents": ["n0"] }] }

// 形态 B：收敛（response_parents 指定 [FINAL RESPONSE] 节点挂到哪些已完成节点上）
{ "finish": "Answer: <short answer>\n<explanation and sources>", "response_parents": ["n0", "n2"] }
```

| 规则                                                                    | 失败行为                          |
| ----------------------------------------------------------------------- | --------------------------------- |
| 字段白名单：只允许 `nodes` / `edges` / `finish` / `response_parents` | `OpsParseError` |
| 扩图的依赖写在 `nodes[i].parents`（正常节点同样要写）；`response_parents` 仅与 `finish` 同用 | `OpsParseError` |
| `nodes` 与 `finish` **互斥**；两者同时出现时保留 `nodes`、丢弃 `finish` | 记 `Ops.warnings` |
| `finish.response_parents` 必须是**已完成**节点；非法时先回灌一次，仍非法则退回连接全部已完成节点 | 回灌 / 自动连接 |
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
3. **context 语义**：固定为 `"parent answer(s)" + "\n---\n" + "Root question: ..."`，**不含**兄弟节点与全部历史。

**构造入口（v1.7 新增，B 实现，A 只调用）**

A 的端到端脚本只通过下面这个工厂拿 searcher，**不直接依赖 `backends/`**——`backends/` 只放搜索/抓取原语，属实现细节：

```python
# lychee_mas/methods/build/mindsearch/searcher.py（与 WebSearcher 同文件，B 拥有）
def make_web_searcher(*, llm, engine: str = "tavily", topk: int = 3,
                      max_queries: int = 3, max_read: int = 3) -> WebSearcher: ...
```

- `engine`：`"tavily"`（默认，唯一已实现）；`"ddg"` / `"wiki"` 为预留名，未实现时显式
  `ValueError`（不静默降级到其他后端）；
- **Tavily 为选定后端**：密钥走环境变量 `TAVILY_API_KEY`（不写进配置、不进 `run_config.json`）；搜索与抓取原语放在 `backends/`（B 实现），工厂负责注入；
- 若后端返回 `raw_content`（Tavily 可一次带回正文），工厂用它充当 `fetch` 结果，`pages_read` 记实际使用条数；
- 未实现时**函数不存在**，`scripts/run_mindsearch.py` 会 `SystemExit` 并提示"先用 `--searcher stub`"——不做隐式回退；
- `engine` 会被写入 `run_config.json`，报告据此声明检索后端。

**Tavily 实现要求（v1.8，B）**

1. `search(queries, topk)`：Tavily 一次只接受一个 query，内部循环；每个 query `max_results=topk`；
2. `fetch(url)`：优先用 Search 返回的 `raw_content`，为空再走 Tavily Extract（markdown）或 HTTP 抓取；
3. **正文截断 6000–8000 字符**（Extract 单页可达 17k），否则摘要 prompt 会爆、token 失控；
4. 失败语义：网络/限流由 `WebSearcher` 转成 `status="error"`；`run()` 内部最多重试 1 次；
5. 密钥只读 `TAVILY_API_KEY`：不写日志、不进 `run_config.json`。

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
| `citation_ok`                        | bool           | 答案引用是否可追溯到检索 sources（v1.5）              |
| `citation_unknown`                   | list[str]      | 未被检索到的引用 URL（v1.5，配合 citation_ok=False）  |

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

### 3.7 规划节点工厂（`planner.py`，签名冻结、已落地）

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
# 脚本层（A 负责：scripts/run_mindsearch.py）读取短答案的方式
prediction = state["answer"]              # 打分口径
prediction_raw = state["outputs"]["planner"]   # 长答案，落盘留档
format_ok = state["format_ok"]            # 聚合为 format_ok_rate
```

真实运行（Tavily 检索 + 硅基流动模型）：

```bash
export LYCHEE_LLM_BASE_URL=https://api.siliconflow.cn/v1
export LYCHEE_LLM_MODEL=deepseek-ai/DeepSeek-V4.1-Flash
export LYCHEE_LLM_API_KEY=<siliconflow key>
export TAVILY_API_KEY=<tavily key>

PYTHONPATH=src python scripts/run_mindsearch.py --arm mindsearch --searcher real --n 1
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

### 6.1 三臂冒烟顺序（A；B 合入前后都能跑）

**阶段 0（不依赖 B）**

```bash
# ① 无检索臂：只验 LLM 通道 + 判分 + 产物
PYTHONPATH=src python scripts/run_mindsearch.py --arm no_search --question "法国的首都是哪里？"
# 期望：stop_reason=no_search，n_nodes=0，metrics.accuracy=null（adhoc 无 gold）

# ② 单轮臂（stub 证据）：应"拒答"而不是编造
PYTHONPATH=src python scripts/run_mindsearch.py --arm single_turn --searcher stub \
  --question "2024 年诺贝尔物理学奖授予了谁？"
# 期望：prediction 说明证据不足；tokens 非 0（证明真的走了一次 LLM 作答）

# ③ 规划臂（stub）：验 DAG/收尾/引用门禁
PYTHONPATH=src python scripts/run_mindsearch.py --arm mindsearch --searcher stub --n 3
# 期望：n_nodes>=2；stop_reason ∈ {finish, redundant, max_turns}；
#       citation_ok 有值（stub 来源下多为 false，属预期）
```

**阶段 1（B 合入 `make_web_searcher` + Tavily 后）**

```bash
# ④ 先 1 题，验通路与引用
PYTHONPATH=src python scripts/run_mindsearch.py --arm mindsearch --searcher real --n 1
# 期望：run_config.engine=tavily；citation_ok=true；n_nodes 2–8；tokens 9k–15k

# ⑤ 再 1 题单轮臂
PYTHONPATH=src python scripts/run_mindsearch.py --arm single_turn --searcher real --n 1

# ⑥ 三臂 × 3 题对照，指标无异常后再扩到 30 题
```

### 6.2 测试归属

| 测试 | 归属 | 是否联网 |
| --- | --- | --- |
| `tests/test_mindsearch_{dag,interpreter,planner,citation,builder,answer}.py` | A | 否 |
| `tests/test_run_mindsearch_wiring.py`（工厂参数透传 + stub 端到端） | A | 否 |
| `tests/test_mindsearch_searcher_factory.py`（工厂 / 未知 engine / 缺密钥） | B | 否 |
| `tests/test_tavily_backend.py`（响应映射 / 限流 / 超时 / raw_content） | B | 否（mock） |
| 真实集成冒烟（`skipif` 无 `TAVILY_API_KEY`） | B | 是，默认跳过 |

---

## 7. 版本记录

| 版本 | 变更                                                                                                                               | 破坏性 |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------- | ------ |
| v1   | 首次冻结：`TaskDAG` / ops 协议 / `SearcherResult` / `MindSearchState` / builder                                                    | —      |
| v1.1 | searcher 提示移出 `prompts.py` → `searcher_prompts.py`（B 独占）                                                                   | 否     |
| v1.2 | `state` 新增 `answer` / `format_ok`；新增 `answer_format.format_answer()`；登记 `planner_node_factory` 签名；补全 §3.2–§3.7、§4–§6 | 否     |
| v1.3 | `TaskDAG.from_dict()` 落地（检查点/续跑）；`ready()` 的 `failed` 终态随 planner 实现 | 否     |
| v1.4 | `state.stop_reason`；`ready()` 语义不变（failed 阻塞子节点） | 否 |
| v1.5 | finish 合法性约束：`min_nodes_before_finish` + 连续拒绝后兜底建节点；`citation_check` 引用级校验（不通过则拒绝并重答）；解析器容忍 nodes+finish 并存 | 否 |
| v1.6 | `spec.system_prompt` 接入 `build_plan_prompt`（MASPO 改写才真正生效）；子问题归一化去重 + `redundant` 收敛；`prompts.py` 与当轮 JSON 格式分离 | 否 |
| v1.7 | 新增真实检索构造入口 `make_web_searcher`（`methods/build/mindsearch/searcher.py`，B 实现）；`backends/` 退为内部原语，脚本不再直接依赖 | 否 |
| v1.8 | 真实检索后端定为 **Tavily**：`make_web_searcher` 的 `engine` 默认 `tavily`，密钥 `TAVILY_API_KEY`；搜索/抓取原语落 `backends/`（B 实现），工厂注入 | 否 |

---

## 8. 与论文的偏差（须写入 PR 与报告）
- **响应节点**：论文由 WebPlanner 通过 `generate_response` 创建名为 `response` 的节点并指定依赖；
  本实现沿用既有命名 `[FINAL RESPONSE]`（语义一致），由 `finish.parents` 给出依赖，
  缺省时连接全部已完成节点。
- **多父节点**：论文 `add_node(query, deps)` 本身允许任意多依赖，本实现一致；但增加了三条工程约束：
  依赖去重、失败父节点的子节点在摘要中标为 `blocked←<id>`、单节点上下文总量按 2400 字符预算均分。


| 项       | 论文                                     | 本实现                         | 原因                                                            |
| -------- | ---------------------------------------- | ------------------------------ | --------------------------------------------------------------- |
| 规划动作 | Python 代码 + 解释器（code as planning） | JSON ops + 严格校验 + 错误回灌 | 无 `exec` 安全面；错误可定位并反馈（论文附录 E 的等价物）       |
| 执行图   | 运行时动态扩展的 DAG                     | 静态壳 + 节点内 DAG            | `extract_view` 只接受静态 DAG / 唯一终端，保住 prerun 能力      |
| 检索器   | WebSearcher（Bing）                      | WebSearcher（Tavily Search/Extract） | 引擎可替换：search/fetch 为注入回调，未知 engine 显式报错       |
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

#### 可选运行统计（2026-11-10，兼容性增量）

MindSearchState和逐题产物增加`planner_prompt_tokens`、`planner_completion_tokens`、
`searcher_prompt_tokens`、`searcher_completion_tokens`、`queries_count`、`pages_read`。
原有总token字段及Searcher/DAG冻结签名不变。统计包含失败尝试已返回的usage；
queries_count为累计查询批次条目数，pages_read为累计取得非空正文次数，均不跨尝试去重。
超时无usage的费用及旧checkpoint历史细分不能反推，完整口径见mindsearch-searcher.md。
