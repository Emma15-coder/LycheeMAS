# MindSearch离线答案的LLM评判

脚本位于`scripts/judge_mindsearch.py`，复用现有OpenAI兼容后端及项目.env配置。
框架`src/lychee_mas/eval/`不新增、不修改。默认只预览；--execute才付费调用模型。
不重新检索或答题，不评价来源有效性和信息完整性，只评价短答案相对题目与参考答案的语义正确性。

先预览（可以传入多个运行目录）：

```bash
PYTHONPATH=src python scripts/judge_mindsearch.py \
  runs/eval_benchmarks/bamboogle/no_search/20261111-164348 \
  runs/eval_benchmarks/bamboogle/single_turn/20261111-164426 \
  runs/eval_benchmarks/bamboogle/mindsearch/20261111-165007
```

真实评判时增加--execute，并建议明确指定新的--out-dir。
默认最多15条记录、15次请求，每次max_tokens=512、timeout=30秒、temperature=0；
--model/--base-url/--api-key-env可覆盖复用的模型配置，不在命令行提供Key值。
SDK自动重试和重定向关闭；已有输出目录拒绝覆盖，重复运行也不会自动恢复并再次计费。
接口是否严格遵循max_tokens以服务实际行为为准，保存服务报告的usage。

同一次运行内，对问题、参考答案、待评答案三项完全相同的输入共享一次判定；
缓存键不包含方法名，Judge只看到上述三项，使用新的独立对话，不看到旧得分或其它方法答案。
失败结果也缓存而不重试，报告标明缓存条数；这不构成独立重复实验或方差估计。

产物（权限600，写入runs/，不覆盖原实验）：

- judge_config.json：提示词、模型、预算、代码版本/未提交状态、输入文件及脚本哈希。
- requests.jsonl：每次真实请求的脱敏消息、原始返回、finish_reason、usage、耗时、错误类别。
- judgments.jsonl：每条答案的判定、EM分数、来源运行、缓存关系和人工复核标记。
- summary.json：按方法汇总正确/错误/不确定/请求或解析错误、coverage和token。
- review.json：不确定、错误、缺gold或EM与Judge分歧的待复核记录。

只有合法JSON且finish_reason=stop才接受判定。错误不算incorrect；
judge_accuracy_decided只以correct+incorrect作分母，必须同时报告decision_coverage。
全部eligible样本的正确率下/上界由accuracy_lower_bound/accuracy_upper_bound给出；
没有gold的记录不调用模型。未返回usage的请求可能计费但用量未知。

规范化EM保留，不能用Judge分数覆盖EM。日期顺序、确切同义名、无矛盾的补充可判correct，
仅年份一致、错误实体或矛盾替代答案不能判correct；歧义标uncertain并人工复核。
Judge可能误判；同模型答题和判分存在偏差，不应宣称复现论文GPT-4o judge的绝对结果。
