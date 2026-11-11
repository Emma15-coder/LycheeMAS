"""离线格式兼容矩阵：区分已支持、保守拒绝和不得接受的事实改写。"""
import asyncio
import json

import pytest
from lychee_mas.methods.build.mindsearch.page_content import resolve_evidence_quote
from lychee_mas.methods.build.mindsearch.searcher import WebSearcher

SUPPORTED = [
    ('plain', 'Born in Scotland.', 'Born in Scotland.'),
    ('bold_star', 'Born in **Scotland**.', 'Born in Scotland.'),
    ('bold_underscore', 'Born in __Scotland__.', 'Born in Scotland.'),
    ('link', 'Born in [Scotland](https://example.org/place).', 'Born in Scotland.'),
    ('bold_link', 'Born in **[Scotland](https://example.org/place)**.', 'Born in Scotland.'),
    ('link_bold_label', 'Born in [**Scotland**](https://example.org/place).', 'Born in Scotland.'),
    ('nbsp', 'Born in\u00a0Scotland.', 'Born in Scotland.'),
    ('narrow_nbsp', 'Born in\u202fScotland.', 'Born in Scotland.'),
    ('heading_substring', '## Born in Scotland.', 'Born in Scotland.'),
    ('list_substring', '- Born in Scotland.', 'Born in Scotland.'),
    ('blockquote_substring', '> Born in Scotland.', 'Born in Scotland.'),
    ('table_cell', '| Birthplace | Scotland |', 'Scotland'),
]
UNSUPPORTED = [
    ('italic_star', 'Born in *Scotland*.', 'Born in Scotland.'),
    ('italic_underscore', 'Born in _Scotland_.', 'Born in Scotland.'),
    ('bold_italic', 'Born in ***Scotland***.', 'Born in Scotland.'),
    ('inline_code', 'Born in `Scotland`.', 'Born in Scotland.'),
    ('strikethrough', 'Born in ~~Scotland~~.', 'Born in Scotland.'),
    ('html_entity', 'Company: A &amp; B.', 'Company: A & B.'),
    ('html_tag', 'Born in <strong>Scotland</strong>.', 'Born in Scotland.'),
    ('soft_line_break', 'Born in\nScotland.', 'Born in Scotland.'),
    ('table_rewrite', '| Birthplace | Scotland |', 'Birthplace: Scotland'),
    ('reference_link', 'Born in [Scotland][place].', 'Born in Scotland.'),
    ('unpaired_bold', 'Born in **Scotland.', 'Born in Scotland.'),
    ('escaped_marker', r'Born in \**Scotland**.', 'Born in Scotland.'),
]


@pytest.mark.parametrize('name,text,quote', SUPPORTED, ids=[x[0] for x in SUPPORTED])
def test_supported_format_maps_to_contiguous_original(name, text, quote):
    original = resolve_evidence_quote(quote, text, text)
    assert original is not None and original in text


@pytest.mark.parametrize('name,text,quote', UNSUPPORTED, ids=[x[0] for x in UNSUPPORTED])
def test_unsupported_rendering_is_conservatively_rejected(name, text, quote):
    assert resolve_evidence_quote(quote, text, text) is None
    # 不支持渲染转换时，模型仍可引用精确原文，不会整页作废。
    assert resolve_evidence_quote(text, text, text) == text


@pytest.mark.parametrize('text,quote', [
    ('Born in **Scotland**.', 'Born in England.'),
    ('Founded in **1812**.', 'Founded in 1813.'),
    ('Value: **-5**.', 'Value: 5.'),
    ('The result is **not approved**.', 'The result is approved.'),
    ('Rate: **3.14**.', 'Rate: 314.'),
    ('Name: alpha_beta.', 'Name: alphabeta.'),
    ('Multiply 2 * 3.', 'Multiply 2  3.'),
    ('Born in [England](https://example.org/Scotland).', 'Born in Scotland.'),
    ('Born in [England](https://example.org "Scotland").', 'Born in Scotland.'),
    ('Born in **Scotland**.\n\nYear: 1881.', 'Born in Scotland. Year: 1881.'),
])
def test_rendering_never_changes_facts_or_promotes_hidden_link_metadata(text, quote):
    assert resolve_evidence_quote(quote, text, text) is None


@pytest.mark.parametrize('name,text,quote', [SUPPORTED[1], UNSUPPORTED[0], UNSUPPORTED[2],
                                           UNSUPPORTED[3], UNSUPPORTED[5]])
def test_formatting_matrix_through_searcher_status_and_usage(name, text, quote):
    outputs = iter([{'queries': ['test']}, {'source_ids': ['S1']},
                    {'answer': 'Scotland [S1]', 'source_ids': ['S1'],
                     'evidence': [{'source_id': 'S1', 'quote': quote}]}])

    async def model(prompt):
        return json.dumps(next(outputs))

    async def search(queries, topk):
        return [{'url': 'https://example.org/evidence', 'title': 'test', 'snippet': 'test'}]

    async def fetch(url):
        return text

    result = asyncio.run(WebSearcher(model, search, fetch).run('Where was the person born?'))
    assert result.pages_read == 1
    if name == 'bold_star':
        assert result.is_ok()
    else:
        assert result.status == 'error' and result.sources == [] and result.answer == ''
        assert 'reason=unsupported_evidence' in result.error
