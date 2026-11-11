# MindSearch 复现 · 评估协议 v1

> 冻结日期：2026-11-02 · Owner：A（B review）· 改动需双方确认
> 原则：**只做同设置内部对比**，不宣称与论文表 1 数值逐点一致（见 §7 偏差表）。

---

## 1. 目标与判定

**要回答的问题**：多智能体「动态分解 + 并行分层检索」是否优于「单轮搜索直接回答」。

| 判定     | 标准                                                                   |
| -------- | ---------------------------------------------------------------------- |
| 主判定   | 每个数据集上 `mindsearch ≥ single_turn_search ≥ no_search`（accuracy） |
| 成本判定 | 同时报告 token / 时延 / 检索次数，说明增益的代价                       |
| 反例接受 | 若某集不成立，如实报告并给原因（如检索质量、题目分布），不隐藏         |

**不做宏观平均**：Bamboogle / Musique / HotpotQA 分别报，禁止用"三集平均"掩盖差异。

---

## 2. 数据集与任务名

| 注册名      | 说明       | 分组别名（照 GAIA level 的写法）                      |
| ----------- | ---------- | ----------------------------------------------------- |
| `bamboogle` | 2 跳组合题 | —                                                     |
| `musique`   | 多跳问答   | `musique_2hop` / `musique_3hop` / `musique_4hop`      |
| `hotpotqa`  | 多跳问答   | `hotpotqa_easy` / `hotpotqa_medium` / `hotpotqa_hard` |

- 另加**单轮检索基线**所需的同数据集（同 split、同抽样），仅换执行臂，不换数据。
- split / dataset id / 下载源以 `eval/benchmarks/source_catalog.json` 登记为准，并写入 `run_config.json`。
- 分组通过**同一 loader + 过滤**实现（不要写 7 个 loader）。

---

## 3. 打分口径（关键决策）

| 数据集      | kind                        | 说明                                      |
| ----------- | --------------------------- | ----------------------------------------- |
| `bamboogle` | `normalized_em`（loader kind仍为exact） | 短答案规整后完全相等 |
| `musique`   | `normalized_em`             | 同上                                      |
| `hotpotqa`  | `normalized_em`             | 论文用GPT-4o judge；本版EM为主，LLM Judge作补充 |

由`scripts/mindsearch_metrics.py`复用框架`_norm`（小写、去ASCII标点、去英文冠词、压空白），
再判定完全相等；不采用框架原exact的最后数字或子串匹配。框架metrics.py保持不变。
空gold不评分；归一化后为空的gold不判正确。不额外做日期重排、别名或单位语义等价。
run_config、逐题记录、metrics及framework质量指标标记score_metric=normalized_em；
kind/scorer_kind=exact保留为原有二值任务分类，不代表调用旧评分函数。
原始结果不覆盖，可使用scripts/rescore_mindsearch.py离线另存重评分；人工复核另列，
不得把人工语义正确率与EM混为同一指标。F1/人工完整性评价尚需独立补充。

**主评分保持离线**：规范化EM可离线复算；LLM Judge作为独立补充，需要额外API费用，
保存提示词、原始判定及usage，结果不得覆盖EM。通用分析脚本仍采用框架原评分，
本实验的EM重算应使用scripts/rescore_mindsearch.py。

**配套约束**：planner 的 `finish` 必须首行为 `Answer: <短答案>`（实体 / yes / no），
其余为解释——这样 `exact` 才能工作。这条写进 `PLANNER_PROMPT`。

> 补充Judge位于scripts/judge_mindsearch.py，不改框架eval目录；默认预览，显式--execute才请求模型。
> 输出correct/incorrect/uncertain，调用或解析失败单独记录，正确率须同时报告判定覆盖率。
> 同模型评判偏差与人工复核限制须注明，使用方法见docs/mindsearch-judge.md。

---

## 4. 规模与抽样

| 项         | 值                                                                            |
| ---------- | ----------------------------------------------------------------------------- |
| 每题集题数 | 30（冒烟 5；全量可选，需在报告标注）                                          |
| 抽样       | 按题号 hash 排序后取前 N，`seed=0`，记录所选 id 的哈希                        |
| 重复       | 每题 1 次（`temperature=0`）；如需稳定性，`k=3` + `self_consistency` 单独一组 |

---

## 5. 实验臂

| 臂                   | 定义                              | 实现位置                                    |
| -------------------- | --------------------------------- | ------------------------------------------- |
| `no_search`          | 纯 LLM 直答（无检索工具）         | `scripts/run_mindsearch.py --arm no_search` |
| `single_turn_search` | 调用一次完整Searcher → 作答，内部检索预算与MindSearch一致 | 同上 |
| `mindsearch`         | 本复现（动态 DAG + 并行分层检索） | 同上                                        |
| `react`（可选）      | ReAct 多轮工具调用                | 同上，时间允许再做                          |

固定超参（写入 `run_config.json`，不允许临场改）：

| 参数                                | 值        |
| ----------------------------------- | --------- |
| `max_turns`                         | 10        |
| `max_subq_per_turn`                 | 3         |
| `max_nodes`                         | 40        |
| `max_concurrency`                   | 8         |
| `topk` / `max_queries` / `max_read` | 3 / 3 / 3 |
| 总读页上限                          | 300       |
| `temperature`                       | 0.0       |
| `seed`                              | 0         |

---

## 6. 产物与指标

每次 run 落 `runs/mindsearch/<task>/<arm>/<timestamp>/`：

| 文件                 | 内容                                                                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `predictions.jsonl`  | 每题：`case_id, task, arm, question, gold, prediction, score, is_correct, tokens, latency_s, n_queries, n_pages, n_nodes, dag_depth` |
| `dag/<case_id>.json` | `TaskDAG.to_dict()`（依赖图 + 每节点状态/答案/来源）                                                                                 |
| `dag/<case_id>.mmd`  | mermaid（报告直接用）                                                                                                                |
| `sources.jsonl`      | 全部来源 `{case_id, qid, url, title}`                                                                                                |
| `run_config.json`    | git SHA、分支、模型、引擎、全部超参、抽样哈希、数据集 id/split、时间戳                                                               |
| `metrics.json`       | 聚合结果（见下）                                                                                                                     |

| 指标                                      | 定义                           |
| ----------------------------------------- | ------------------------------ |
| `accuracy`                                | 正确题数 / 总题数              |
| `avg_tokens`                              | 每題 `prompt+completion` 平均  |
| `avg_latency_s`                           | 端到端平均时延                 |
| `avg_queries` / `avg_pages` / `avg_nodes` | 检索与图规模（成本轴）         |
| `dag_depth`                               | 最长依赖链长度（探索深度代理） |

**pass@K**：`k>1` 时用 `run_processed(method="parallel", k=K)` 采样，打分侧按 case 聚合
（pass@1 = 各 case 均分再平均；pass@K = best-of-K 再平均）。

---

## 7. 与论文的对齐 / 偏差

| 维度     | 论文                                        | 本复现                                      | 状态                     |
| -------- | ------------------------------------------- | ------------------------------------------- | ------------------------ |
| 模型     | GPT-4o / InternLM2.5-7B-chat                | OpenAI 兼容端点（默认 GPT-4o 或等价强模型） | 部分对齐；版本滚动需记录 |
| 检索     | Bing                                        | Tavily Search + Extract                     | 偏差；结果分布不同       |
| 数据     | 全量                                        | 每集 30 题                                  | 偏差；只比趋势           |
| 打分     | Bamboogle/Musique EM；HotpotQA GPT-4o judge | 统一 `exact`（+f1）                         | 偏差；绝对值不可比       |
| open-set | 100 条 × 5 专家盲评                         | 20–30 条 × LLM-judge 或 2 人盲评（三轴）    | 简化；仅定性佐证         |
| 超参     | max_turns 10 等                             | 同左                                        | 对齐                     |

> 报告必须包含本节；任何"和论文一样"的表述都要附对应的对齐证据。

---

## 8. 复算与验收

```bash
# 推理（落盘，不打分）
PYTHONPATH=src python scripts/run_mindsearch.py \
  --arm mindsearch --benchmark bamboogle --n 30

# 事后打分（离线，可复算）
PYTHONPATH=src python scripts/analyze_mindsearch_run.py runs/mindsearch/... --score-predictions
```
