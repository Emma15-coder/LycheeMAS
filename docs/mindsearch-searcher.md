# MindSearch Searcher（独立检索模块）

不含 Planner；保持 IFACE v1.3 的 WebSearcher 构造签名、run 签名和返回字段。

## 流程与约束

1. 注入的模型改写查询（JSON queries），去重并限制 max_queries。
2. search(queries, topk) 返回 url/title/snippet；规范化 URL，合并去重。
3. 模型按候选 ID 选页，最多 max_read 次 fetch；对有界正文离线筛选原文片段，
   不把摘要当作成功抓取的正文。
4. 模型仅根据正文输出 answer/source_ids/evidence。检查引用 ID、每个来源的逐字摘录。
   结果只保留实际读取且引用的 SourceRef；答案中的 [S1] 对应返回 sources[0]，依次类推。

来源有效和摘录存在是程序校验；摘录是否足以支持答案仍属于语义判断，不能保证模型绝无幻觉。
网页与 context 在提示中标记为不可信数据。context 仅传入父节点答案 + `\n---\n` + 根问题，
不添加兄弟节点或历史；不将 context 作为新检索证据。

- ok：有非空正文、有效答案及引用；empty：无候选、未选页、正文为空或证据不足。
- error：请求/抓取失败、超时、JSON或引用校验失败；不返回伪答案，也不回显异常原文。
- 本版**内部重试0次**（符合最多1次的冻结上限），SDK及HTTP适配器也不重试。
- 每次 run 至多3次 LLM；Tavily至多 max_queries 次 Search、max_read 次 Extract。
- max_queries 1–3；topk/max_read 1–10。每页抓取正文最多300000字符、提供给模型最多12000字符；每条查询256字符，
  子问题/context各8000字符；模型响应32000字符。超长输入报错，正文/结果受限截断。
- 每个回调35秒、run等待上限180秒；Tavily传输20秒，HTTP响应最多2MB。
  回调取消不能强行终止已运行的工作线程，线程仍受传输超时限制；这些不是硬实时保证。
- pages_read 计成功取得非空正文的页面；失败保留此前计数。
  token统计包含所有已收到的模型返回（包括格式非法返回）；裸字符串回调无usage计0。
  超时/断线时服务端可能已计费但客户端无usage，无法计入本地统计。
- Tavily仅使用 Search basic（include_answer=false、auto_parameters=false）及 Extract basic，
  不使用 Research。HTTP 200的 failed_results 仍视为失败；不尝试其它链接掩盖失败。

官方请求依据：[Search](https://docs.tavily.com/documentation/api-reference/endpoint/search)、
[Extract](https://docs.tavily.com/documentation/api-reference/endpoint/extract)。

## 环境与单子问题冒烟

仅需要现有项目环境及 `openai/httpx`，Tavily用标准库，不需要 SDK/MCP。
缺依赖时按需 `uv pip install -e '.[searcher]'`，不升级其它依赖。
YAML仍为配置模板，不自动加载；独立入口显式组装后端。
默认模型端点为 `https://api.siliconflow.cn/v1`，模型为
`deepseek-ai/DeepSeek-V4.1-Flash`；可用 `LYCHEE_LLM_BASE_URL/LYCHEE_LLM_MODEL` 覆盖。
不额外传递未经确认的思考开关。

在项目根目录用 Bash 执行；Key仅通过临时环境传入，不保存、不打印：

```bash
(
  set +x
  read -r -s -p 'SiliconFlow Key（隐藏）: ' SILICONFLOW_API_KEY
  printf '\n'
  read -r -s -p 'Tavily Key（隐藏）: ' TAVILY_API_KEY
  printf '\n'
  export SILICONFLOW_API_KEY TAVILY_API_KEY
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python scripts/smoke_mindsearch_searcher.py
)
```

固定上限：1个子问题，3次模型调用（改写/选页/总结，各最多512输出tokens），
1次Tavily Search（最多3条），1次Extract；不重试、不自动重定向，不落盘。
输出结果JSON含token/页数/查询/来源，另输出总耗时；非ok退出码1。
模型若输出不合法JSON或无依据引用，会明确返回error，不额外花费请求修复。

## 交给 Planner

```python
from lychee_mas.backends.async_chat import AsyncChat
from lychee_mas.backends.tavily_backend import TavilyBackend
from lychee_mas.methods.build.mindsearch.searcher import WebSearcher

# backend为调用方创建的OpenAICompatibleBackend，传输timeout=30且关闭SDK重试。
# 独立脚本 make_searcher() 展示了完整客户端创建和关闭方式。
tavily = TavilyBackend()
searcher = WebSearcher(AsyncChat(backend, max_new_tokens=512),
                       tavily.search, tavily.fetch,
                       max_queries=1, topk=3, max_read=1)
result = await searcher.run(sub_question, parent_answer + "\n---\n" + root_question)
if result.is_ok():
    dag.mark_done(qid, result.answer, [s.to_dict() for s in result.sources])
elif result.status == "empty":
    # 交给Planner决定重规划/mark_failed，不把空结果当作有依据回答。
    pass
else:
    dag.mark_failed(qid, result.error)
```

并发run的结果和统计互相隔离。服务回调须可并发使用；模型阻塞生成由AsyncChat移入线程。
调用方应在全部任务（含取消后的传输收尾）结束后关闭模型客户端。

## 离线验证

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_mindsearch_*.py
.venv/bin/ruff check src scripts/smoke_mindsearch_searcher.py tests/test_mindsearch_searcher.py tests/test_mindsearch_backends.py
```

新增测试只用假模型、假搜索、假正文/HTTP，不读真实Key，不发网络请求。

## empty 诊断（保持相同请求预算）

独立冒烟入口现默认在终端 stderr 打印经过脱敏的 JSON 阶段日志；不保存文件。
库内使用专用 logger `lychee_mas.mindsearch.trace`，默认不打开 INFO 日志。
日志有 run_id，避免并发任务混淆；不记录认证头或异常原文。
输出模型正文和网页前500字符可能含业务数据，分享日志前仍应自行检查。

- `search.results`：返回条数、有效URL去重后数量及候选页。
- `pages.selected`：实际采用的来源ID与URL（敏感查询参数脱敏）。
- `tavily.extract`：API返回的完整正文字符数、适配器截断后返回字符数。
- `fetch.content`：Searcher收到的正文长度、截断后长度和前500字符。
- `summary.input`：实际提供给总结模型的每页长度。
- `model.response`：rewrite/select/summarize每次响应正文、finish_reason、token和耗时。
  旧回调没有finish_reason时记录null；现有API后端已保留该字段。
- `searcher.empty`：明确记录以下分支原因：
  `no_valid_search_results`、`no_pages_selected`、`no_usable_page_content`、
  `summary_declared_insufficient_evidence`。
- `searcher.end`：最终状态、页数、token和总耗时。

`pages_read`只在成功获得非空字符串时加一，不代表网页相关、事实齐全或被答案引用。
因此原冒烟若`empty + pages_read=1`，唯一可达分支是总结模型返回空answer、空source_ids、
空evidence；旧日志不能反推出所选网页、截断情况或模型finish_reason。
不能仅凭这个状态断言搜索质量、正文截断或输出长度就是根因。

补充修复：若finish_reason为length/content_filter，即便正文恰好是合法的空JSON，
也返回error并记录model.incomplete，避免把不完整/受限输出当作正常的证据不足。
此改动不放宽任何引用或摘录校验，不增加重试或请求次数。

需要复查真实请求时，只执行上面的同一道默认题命令一次，不循环运行。
返回诊断输出后再判断是否需调整选页或截断策略；当前不做猜测性放宽。

## 本地持久密钥与运行包装（推荐）

一次性在终端执行：

```bash
.venv/bin/python scripts/run_searcher_smoke.py setup
```

Python getpass分别隐藏输入两份Key；若终端不支持隐藏输入则拒绝退化为明文。
原子保存至项目根目录 `.env`，文件权限600；`.gitignore`已忽略该文件。
项目根目录保留原权限，但不允许组或其他用户写入，不会改成700。
setup再次执行会原子替换旧Key；输入无效则保留旧文件。文件不包含export指令，
包装器只解析两项字面量KEY=value，不source、不执行shell扩展，不复制到仓库。

```bash
# 只检查格式/所有权/权限，不调用服务、不显示Key
.venv/bin/python scripts/run_searcher_smoke.py check

# 加载密钥，固定执行已授权的同一道单题冒烟
.venv/bin/python scripts/run_searcher_smoke.py
```

包装器检查项目目录不可被组/其他用户写入、文件600和当前用户所有权，拒绝符号链接；
Key只通过子进程环境传递。
不接受任意命令、题目或批量参数，强制使用已验证的硅基流动端点与模型。
保留原预算：模型最多3次、各最多512输出tokens，Tavily Search 1次、Extract 1次，零重试。
可由编码代理直接通过包装脚本运行这一已授权的单题验证，无需再次粘贴Key命令；
每次运行仍应报告状态、token、耗时。批量评测需另行说明费用和范围。
包装器不会自动循环执行、启动后台任务或修改Git。

密钥文件属于本机持久配置，删除该文件即可移除本地保存的Key；勿粘贴其内容到聊天。


## 长网页正文预算修复（离线验证）

旧实现把Tavily正文在适配器内直接截成前12000字符，Searcher又取前12000字符。
这会让后部证据永久丢失。现将两个预算分开：API响应仍最多2MB，适配器返回最多300000字符，
Searcher在该范围内进行确定性的原文筛选，最终每页模型输入仍不超过12000字符。
如果显式设置较小的Tavily max_page_chars，后部材料仍不可恢复。

`page_content.py`不使用模型知识或额外请求：
- 跳过独占行的Markdown链接导航；含文字说明、表格字段或代码块的行保留。
- 用子问题及改写查询的英文词/中文二元组与原文块做覆盖排序，优先选命中块和邻块。
- 按原文顺序返回选中片段；不足预算时补充前部上下文。没有匹配时保留清理后的开头。
- 输出原文区间，摘要中的摘录必须同时存在于选中片段和有界原文中，
  防止跨片段拼接产生原文没有的“引用”。来源和摘录校验没有放宽。

fetch.content日志新增scanned_chars、navigation_chars、selected_spans和strategy；
content_preview现在显示实际供给模型的内容开头。tavily.extract仍记录原始和返回长度。
pages_read含义不变：成功收到非空正文的页数；如果清理后仅剩导航，可能pages_read=1但
返回no_usable_page_content。旧冒烟的“pages_read=1只能是总结空答案”结论仅适用于修复前逻辑。

离线测试覆盖长导航前缀、长无关正文前缀、中英查询、相邻事实、预算、原文区间、
跨片段伪引用和保持原请求次数的完整流程。该启发式不能保证覆盖所有事实；
真实页面是否因此改善，仍需单独一次有预算的真实冒烟验证。

## 完整捕获、离线重放与本次诊断结论

`scripts/diagnose_mindsearch_searcher.py`是显式诊断入口，不由普通冒烟自动启用。
它读取同一项目根目录.env中的密钥，保存完整脱敏prompt、模型原始正文/finish_reason/usage、
搜索结果、fetch返回正文、结果和调用计数到指定文件。文件600，目录700；
诊断文件放在被Git忽略的`runs/searcher_debug/`，不包含认证头，禁止提交这些材料。
中断/超时前已捕获的材料仍保留；无模型响应时不编造usage。

```bash
# 离线重放：不加载Key，不发请求；代码/提示必须与捕获版本一致。
PYTHONPATH=src .venv/bin/python scripts/diagnose_mindsearch_searcher.py replay runs/searcher_debug/round1.json
```

提示或算法变更后，严格重放会因prompt不同而失败，这是预期的版本不一致检测。
本次round1是在修改前完成逐字重放确认的；它的完整材料和离线审核保存在runs目录。
支持`--reuse-web <capture>`复用相同query/URL的搜索抓取；缓存未命中立即停止，不悄悄追加请求。
run模式始终最多3次模型、1次Search、1次Extract，不重试；已有输出文件时拒绝覆盖重跑。
真实调用必须遵循当前用户授权的题目/轮数，不能因存在此入口就自行扩大评测。

具体发现：旧的词项覆盖排序偏好重复主体名称、其他家属的“出生”段落和参考资料，
捕获的11994字符片段没有保留主体“生于华沙”的句子，但完整249269字符原文有。
出生表述使用历史政区，未建立“现属国家”关系，不能用人物国籍代替城市现属国家。
来源ID对应正确，summary的证据不足约束没有冲突，不能通过削弱约束强行输出答案。

修复：改写/选页提示要求覆盖全部关系和时间限定；国际实体优先英文查询。
正文排序用可见文字而非Markdown URL属性，保留早期实质正文/信息表，并用词项稀有度和
边际覆盖降低重复主体名的收益。未硬编码题目答案、网站或城市词。
模型输入仍每页最多12000字符，摘录仍同时检查选中内容和原始正文。

本次授权的3轮真实调用已结束：第1轮原题复现empty；修复后第2轮在选页API超时；
第3轮不同问题已经抓到并保留明确的铁塔位于巴黎证据，但总结API超时。
因此不能声明真实端到端通过。详情见runs/searcher_debug/report.json；超时请求的服务端
计费/usage未知。后续验证必须另获授权，优先利用这些已捕获材料，不重复抓取已知网页。


## 端到端入口的工厂接线

`methods/build/mindsearch/searcher.py::make_web_searcher`已实现IFACE v1.7约定签名，
同时从`lychee_mas.methods.build.mindsearch`导出：

```python
from lychee_mas.methods.build.mindsearch import make_web_searcher

# llm为调用方提供的异步回调；工厂不创建模型、不发送请求。
searcher = make_web_searcher(llm=llm, engine="tavily", topk=3,
                             max_queries=1, max_read=1)
result = await searcher.run(sub_question, context)
```

工厂惰性创建TavilyBackend，从环境变量TAVILY_API_KEY读取凭据，注入同一个后端的
search/fetch绑定方法；ddg/wiki及未知引擎明确ValueError，不静默回退。
非法预算及不可调用的llm在构造时明确报错。topk/max_read支持1–10，max_queries支持1–3。
`run_mindsearch.py::build_searcher`的real分支现在可直接使用它，不需要修改脚本。

本次只补工厂接线，沿用已验证的原文300000字符/模型输入12000字符上限；
IFACE v1.8提出的6000–8000字符要求与当前实现存在差异，尚待联调时统一。
当前Search请求关闭include_raw_content，正文来自独立Extract，没有可复用的Search正文缓存。
工厂不自动加载本地密钥文件、不修改调用方模型的重试/超时策略；端到端预算需由运行入口控制。

## Planner + Searcher 离线联调

`tests/test_mindsearch_integration.py`通过端到端脚本的build_searcher(real)创建真实WebSearcher，
配合真实planner_node_factory、StateGraph编译执行和run_mindsearch结果适配；
只把LLM回调与Tavily搜索/抓取原语替换为脚本化假实现。不读本地Key、不发送网络请求。

已覆盖：
- 两跳依赖、独立节点并行、并发上限，以及父上下文不包含兄弟结论。
- Searcher的SourceRef转换为DAG来源字典、最终响应依赖和来源汇总、JSON/Mermaid/来源落盘。
- Planner与Searcher共享模型时，所有调用usage精确相加，不遗漏、不重复记账。
- 空正文、模型声明证据不足、搜索失败、抓取超时及非法来源ID。
- 连续失败giveup、轮数上限强制收尾、失败父节点阻塞子节点及替代路径。
- 最终URL引用非法时反馈重答，且不额外检索；全部无来源时同样不能把虚构URL判为有效。

联调修复：check_citations过去在known为空时立即通过，导致模型凭空输出URL仍被判合规。
现仅在known与cited都为空时允许no_sources；有URL但无已知来源返回unknown_urls。
没有修改接口签名、Searcher状态语义、原文摘录校验或模型参数。
给up/预算耗尽的结果仍标记UNRESOLVED；citation_ok只表示URL可追溯性，不证明答案语义正确。

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_mindsearch_integration.py
```

这证明离线集成链路可运行，不代表完整链路真实模型评测已通过。
真实联调前仍需单独设置整题总调用预算、关闭SDK默认重试、加载项目根目录.env中的Key，并核对超时。
`--searcher stub`只替换检索器，运行脚本仍调用真实LLM，不能拿它当作完全离线测试。

## 直接运行 run_mindsearch.py 的API配置

端到端入口现会自动读取脚本所在项目根目录的`.env`，无需source或手动导出Key。
采用python-dotenv的load_dotenv(stream=..., override=False)加载，支持标准.env的
export、引号、注释和${VAR}展开；不执行shell命令。读取前仍校验固定路径、
要求文件是当前用户拥有的600普通文件，拒绝符号链接，错误不回显文件内容。
仅在main执行时加载；导入模块或运行--help不会读取密钥。

默认与当前已验证的服务一致：
- 模型：deepseek-ai/DeepSeek-V4.1-Flash
- 地址：https://api.siliconflow.cn/v1
- 模型Key：SILICONFLOW_API_KEY；搜索Key：TAVILY_API_KEY

配置优先级为CLI参数 > 已有进程环境 > 项目.env > 上述默认值。
模型/地址可由--model、--base-url或LYCHEE_LLM_MODEL、LYCHEE_LLM_BASE_URL覆盖。
Key变量名可由--api-key-env或LYCHEE_LLM_API_KEY_ENV指定；若存在旧的
LYCHEE_LLM_API_KEY则保留兼容，否则默认选SILICONFLOW_API_KEY。
自定义Key名称可放在.env，通过--api-key-env或LYCHEE_LLM_API_KEY_ENV指定；
不自动复制Key或创建别名。
.env不存在时允许完全使用进程环境；若.env存在但权限/内容无效，明确失败。

```bash
# 只检查配置，不创建模型客户端、不访问数据集、不调用服务，不显示Key
PYTHONPATH=src .venv/bin/python scripts/run_mindsearch.py --searcher real --check-config

# 完整链路单题运行（会产生真实调用；规划/检索预算仍应按实验范围设置）
PYTHONPATH=src .venv/bin/python scripts/run_mindsearch.py \
  --arm mindsearch --searcher real --question '埃菲尔铁塔位于哪个城市？' \
  --max-turns 2 --max-nodes 1 --max-subq 1 --concurrency 1 \
  --max-queries 1 --max-read 1 --max-new-tokens 512
```

SDK自动重试关闭；Planner自身的JSON修正/引用反馈仍由规划参数控制，不能把上述参数
误认为整题HTTP硬调用上限。run_config.json只记录Key变量名称和.env是否加载，不保存Key。
标准predictions/config/metrics/sources/DAG产物逻辑保持不变；异常原文不写入产物，避免反射凭据。


API配置加载依赖python-dotenv（已加入config/searcher/all可选依赖）。
权限校验通过已打开的文件读取后，以stream传入load_dotenv，避免校验后按路径重开文件。
进程环境仍优先于.env；CLI参数继续覆盖解析后的配置，--help不加载依赖或密钥。

### 主入口失败诊断

在 `scripts/run_mindsearch.py` 命令中显式增加 `--searcher-trace`，会在本次输出目录
保存权限600的 `searcher_trace.jsonl`。默认不开启。日志按 `run_id` 区分并行子问题，
保存脱敏后的完整Searcher提示词、模型返回及finish_reason、候选/选中URL、
抓取正文（最多300000字符）、提供模型的片段和阶段耗时。
Searcher错误记录增加静态stage/reason，外部异常正文仍隐藏；引用校验和请求预算不变。
日志用于离线复现Searcher，未包含Planner完整交互或失败的HTTP响应正文。

旧运行仅有DAG中的ValueError/TimeoutError时，无法事后确定具体校验分支或超时阶段。
当前单次模型/搜索批次/抓取的等待上限为35秒，Searcher整体180秒；
Tavily每条请求超时20秒，最多3条查询顺序请求。因此搜索批次可能先触发35秒总等待，
也可能在其它阶段超时，不能仅凭子问题内容推断为网页过大。
手输单题无gold时，framework.mean_score=0不表示答案错误，应结合stop_reason和证据审计。

2026-11-10捕获回归：模型可能复制网页可见文字而省略Markdown链接包装，或将不换行空格
复制为普通空格。`resolve_evidence_quote`为这些情况建立确定性字符位置映射，返回连续原文摘录，
仍要求摘录同时存在于提供模型的片段和抓取原文；不允许改词、模糊匹配或跨片段拼接。
`evidence.verified`记录映射前后摘录以供审计。最小样例位于
`tests/fixtures/mindsearch_quote_formatting.json`。

主入口现在将Searcher的模型等待与`--timeout`对齐（默认120秒，最多受180秒整体预算约束）；
搜索/抓取的外层35秒及Tavily单请求20秒保持不变。独立构造WebSearcher的模型等待默认仍为35秒。
这是修复主入口SDK配置120秒、Searcher却提前35秒取消的不一致，并不保证服务一定及时响应。

### 搜索后端错误分类

Tavily使用`SearchBackendError`（兼容RuntimeError）携带固定原因码：`http_error`、
`network_timeout`、`connection_error`、`invalid_json`、`invalid_response`、
`response_too_large`、`extract_failed`、`unexpected_error`。
HTTP错误仅保留数值状态码；不记录响应正文、认证头或网络异常正文。
Searcher将原因码及可选`http_status`写入既有`error`文本和`searcher.error`事件，
因此Planner的失败节点也能保留分类。传输失败另有`tavily.error`阶段耗时记录。
错误分类不是对历史RuntimeError的反向推断，也不增加自动重试或改变预算。

### 逐题成本和检索强度

`predictions.jsonl`新增`planner_prompt_tokens`、`planner_completion_tokens`、
`searcher_prompt_tokens`、`searcher_completion_tokens`、`queries_count`和`pages_read`。
`metrics.json`原有汇总新增这些字段的`total_`及`avg_`统计；`framework`标准块保持原口径。
Planner/state仅增加统计键，不改变SearcherResult或DAG节点结构。

Planner成本包含规划、格式/引用纠错及最终回答；Searcher包含查询改写、选页和证据总结。
新运行中两部分按输入/输出分别相加等于原有总数，失败尝试中已经返回的usage也计入。
未返回usage的超时/断连请求无法计账，这些值不是完整账单。旧结果和旧checkpoint只有总数时，
无法回溯拆分历史消耗；应从新运行开始使用细分数据。

`queries_count`累计Searcher提交给search回调的查询列表长度，包括失败批次和重复尝试，
不做跨节点去重。若批次中途失败，列表后面的查询可能尚未实际发出，因此不能视为HTTP请求数。
`pages_read`累计成功取得非空正文的次数，包含后来总结失败的页面；同一URL读两次算两次，
并非独立网页数、有效证据页数或最终引用数。no_search的两项检索计数为0；
非检索回答的模型消耗归入Planner侧（single_turn同理，包括其最后一次回答）。

### Single-turn基线的检索预算（2026-11-11）

`--arm single_turn`只调用一次完整`WebSearcher.run()`，然后作答，不构建DAG或继续规划。
内部与MindSearch使用相同的`--max-queries`、`--topk`、`--max-read`配置，默认均为3；
一次Searcher调用可包含多条Tavily搜索请求及多页Extract，不等同于一次底层HTTP请求。
此前脚本将三个参数强制改为1，现已移除覆盖，run_config记录值与实际预算一致。
这是项目内部统一预算的单轮基线，不宣称复现论文未明确给出的Single-turn底层请求预算。
旧1/1/1结果保留为历史基线，不与新配置混合汇总。单轮调用仍可能因证据不足返回empty。
