import asyncio
import json

from lychee_mas.methods.build.mindsearch.page_content import select_page_content
from lychee_mas.methods.build.mindsearch.searcher import WebSearcher


def test_navigation_before_article_does_not_consume_model_budget():
    navigation = "* [Contents](#contents)\n" * 1000
    fact = "Marie Curie was born in Warsaw. Warsaw is now in Poland."
    raw = navigation + "\n# Biography\n" + fact
    assert fact not in raw[:12000]  # 原策略确实丢失证据
    result = select_page_content(raw, "Marie Curie birthplace")
    assert fact in result.text and len(result.text) <= 12000
    assert result.navigation_chars == len(navigation)
    assert all(raw[start:end] in result.text for start, end in result.spans)


def test_long_non_navigation_prefix_query_match_near_tail():
    raw = "Unrelated weather report.\n" * 1000 + "Marie Curie was born in Warsaw.\n"
    result = select_page_content(raw, "Marie Curie birthplace")
    assert "Marie Curie was born in Warsaw." in result.text
    assert len(result.text) <= 12000
    assert result == select_page_content(raw, "Marie Curie birthplace")


def test_chinese_query_and_neighboring_country_sentence():
    raw = "这段讲述其他事情。\n" * 2000 + "居里夫人出生在华沙。\n华沙现在属于波兰。\n"
    result = select_page_content(raw, "居里夫人出生在哪个城市，该城市现在属于哪个国家？")
    assert "居里夫人出生在华沙。" in result.text and "华沙现在属于波兰。" in result.text


def test_inline_links_tables_and_code_are_not_navigation():
    raw = ("Curie was born in [Warsaw](https://x.org).\n"
           "| 出生 | [华沙](https://x.org) |\n"
           "```markdown\n[example](https://x.org)\n```\n")
    result = select_page_content(raw, "Curie 出生")
    assert "born in [Warsaw]" in result.text and "| 出生 |" in result.text
    assert "[example](https://x.org)" in result.text


def test_no_match_still_bounded_no_knowledge_added():
    raw = "x" * 13000
    selected = select_page_content(raw, "birthplace").text
    assert 0 < len(selected) <= 12000 and selected == raw[:len(selected)]
    assert select_page_content("* [Menu](#menu)\n", "birthplace").text == ""


def test_scan_limit_and_exact_span_output():
    raw = "x" * 300000 + "Marie Curie was born in Warsaw."
    result = select_page_content(raw, "Marie Curie")
    assert result.scanned_chars == 300000 and "Warsaw" not in result.text
    assert result.text == "\n\n".join(raw[a:b] for a, b in result.spans)


def test_offline_end_to_end_old_empty_new_evidence_no_extra_calls():
    nav = "* [Menu](#menu)\n" * 1000
    fact = "Marie Curie was born in Warsaw. Warsaw is in Poland."
    raw = nav + fact

    async def scenario(body):
        calls = {"model": 0, "search": 0, "fetch": 0}

        async def llm(prompt):
            calls["model"] += 1
            if prompt.startswith("Rewrite"):
                return '{"queries":["Marie Curie birthplace"]}'
            if prompt.startswith("Select"):
                return '{"source_ids":["S1"]}'
            if fact not in prompt:
                return '{"answer":"","source_ids":[],"evidence":[]}'
            return json.dumps({"answer": "Warsaw, Poland [S1]", "source_ids": ["S1"],
                               "evidence": [{"source_id": "S1", "quote": fact}]})

        async def search(queries, topk):
            calls["search"] += 1
            return [{"url": "https://x.org/", "title": "Curie", "snippet": ""}]

        async def fetch(url):
            calls["fetch"] += 1
            return body

        s = WebSearcher(llm, search, fetch, max_queries=1, topk=3, max_read=1)
        return await s.run("Marie Curie birthplace and current country?"), calls

    old, _ = asyncio.run(scenario(raw[:12000]))
    new, calls = asyncio.run(scenario(raw))
    assert old.status == "empty" and new.is_ok()
    assert calls == {"model": 3, "search": 1, "fetch": 1}
    assert new.sources[0].url == "https://x.org/"


def test_fragment_join_cannot_create_new_quote():
    async def llm(prompt):
        if prompt.startswith("Rewrite"):
            return '{"queries":["alpha beta"]}'
        if prompt.startswith("Select"):
            return '{"source_ids":["S1"]}'
        return json.dumps({"answer": "made up [S1]", "source_ids": ["S1"],
                           "evidence": [{"source_id": "S1", "quote": "alpha\n\nbeta"}]})

    async def search(*args):
        return [{"url": "https://x.org/"}]

    async def fetch(url):
        return "alpha\n* [Menu](#menu)\nbeta"

    result = asyncio.run(WebSearcher(llm, search, fetch).run("alpha beta"))
    assert result.status == "error" and not result.answer


def test_navigation_only_is_empty_but_counts_nonempty_fetch():
    async def llm(prompt):
        if prompt.startswith("Rewrite"):
            return '{"queries":["query"]}'
        assert prompt.startswith("Select")  # 不允许调用总结模型凭空作答
        return '{"source_ids":["S1"]}'

    async def search(*args):
        return [{"url": "https://x.org/"}]

    async def fetch(url):
        return "* [Menu](#menu)\n" * 1000

    result = asyncio.run(WebSearcher(llm, search, fetch).run("q"))
    assert result.status == "empty" and result.pages_read == 1
    assert not result.answer and not result.sources


def test_small_budget_keeps_highest_ranked_tail_block():
    fact = "Marie Curie was born in Warsaw."
    raw = "irrelevant\n" * 10000 + fact
    result = select_page_content(raw, "Marie Curie", budget=100)
    assert fact in result.text and len(result.text) <= 100
    assert result.text == "\n\n".join(raw[a:b] for a, b in result.spans)


def test_captured_birth_relation_is_not_replaced_by_child_birth():
    """真实捕获的小段原文；其完整249269字符材料保存在被git忽略的runs目录。"""
    from pathlib import Path

    fixture = json.loads((Path(__file__).parent / "fixtures" /
                          "mindsearch_birth_excerpt.json").read_text())
    assert fixture["old_selected_has_birth_excerpt"] is False
    raw = ("* [Menu](#menu)\n" * 1000 +
           "| 出生 | subject birthplace details |\n" + fixture["birth_excerpt"] + "\n\n" +
           (fixture["selected_distractor"] + "\n") * 1000)
    selected = select_page_content(raw, "居里夫人出生在哪个城市？")
    assert fixture["birth_excerpt"] in selected.text
    assert len(selected.text) <= 12000


def test_captured_historical_birth_does_not_prove_current_country():
    from pathlib import Path

    fixture = json.loads((Path(__file__).parent / "fixtures" /
                          "mindsearch_birth_excerpt.json").read_text())
    calls = []

    async def model(prompt):
        calls.append(prompt)
        if prompt.startswith("Rewrite"):
            return '{"queries":["subject birthplace and current country"]}'
        if prompt.startswith("Select"):
            return '{"source_ids":["S1"]}'
        # 原始模型的真实弃答；测试没有注入“华沙现属波兰”的补充知识。
        pages = json.loads(prompt.split("Input JSON:\n", 1)[1])["pages"]
        assert fixture["birth_excerpt"] in pages["S1"]
        return fixture["original_response"]

    async def search(*args):
        return [{"url": fixture["source"], "title": "biography", "snippet": ""}]

    async def fetch(url):
        return fixture["birth_excerpt"]

    result = asyncio.run(WebSearcher(model, search, fetch).run("出生城市现在属于哪个国家？"))
    assert result.status == "empty" and not result.answer and not result.sources
    assert result.pages_read == 1 and len(calls) == 3


def test_captured_other_question_evidence_passes_strict_quote_validation():
    from pathlib import Path

    fixture = json.loads((Path(__file__).parent / "fixtures" /
                          "mindsearch_eiffel_excerpt.json").read_text())

    async def model(prompt):
        if prompt.startswith("Rewrite"):
            return '{"queries":["Eiffel Tower city location"]}'
        if prompt.startswith("Select"):
            return '{"source_ids":["S1"]}'
        pages = json.loads(prompt.split("Input JSON:\n", 1)[1])["pages"]
        assert fixture["excerpt"] in pages["S1"]
        # 离线假模型：只证明捕获的真实证据能通过引用核验，不冒充真实模型成功。
        return json.dumps({"answer": "Paris [S1]", "source_ids": ["S1"],
                           "evidence": [{"source_id": "S1", "quote": fixture["excerpt"]}]})

    async def search(*args):
        return [{"url": fixture["source"], "snippet": ""}]

    async def fetch(url):
        return fixture["excerpt"]

    result = asyncio.run(WebSearcher(model, search, fetch).run(
        "In which city is the Eiffel Tower located?"))
    assert result.is_ok() and result.answer == "Paris [S1]"
    assert result.sources[0].url == fixture["source"]
