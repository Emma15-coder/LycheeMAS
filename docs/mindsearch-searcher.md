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
原子保存至项目外 `~/.config/lychee-mas/secrets.env`，目录权限700、文件600。
setup再次执行会原子替换旧Key；输入无效则保留旧文件。文件不包含export指令，
包装器只解析两项字面量KEY=value，不source、不执行shell扩展，不复制到仓库。

```bash
# 只检查格式/所有权/权限，不调用服务、不显示Key
.venv/bin/python scripts/run_searcher_smoke.py check

# 加载密钥，固定执行已授权的同一道单题冒烟
.venv/bin/python scripts/run_searcher_smoke.py
```

包装器检查目录700、文件600、当前用户所有权，拒绝符号链接；Key只通过子进程环境传递。
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
它读取同一项目外密钥，保存完整脱敏prompt、模型原始正文/finish_reason/usage、
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
