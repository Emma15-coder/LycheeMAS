"""有界、确定性的原文片段筛选；不调用模型、不推断或改写事实。"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

# 仅识别独占行的Markdown链接导航；含正文文字/表格字段的行仍保留。
_LINK = re.compile(r"!?\[[^\]\n]*\]\([^\n)]*(?:\([^\n)]*\)[^\n)]*)*\)")
_WORD = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]+", re.IGNORECASE)
_STOP = {"a", "an", "the", "is", "are", "was", "in", "of", "and", "where", "what", "which"}


@dataclass(frozen=True)
class PageContent:
    text: str
    spans: list[tuple[int, int]]
    scanned_chars: int
    navigation_chars: int


def _terms(text: str) -> set[str]:
    terms = set()
    for word in _WORD.findall(text.lower()):
        if "\u3400" <= word[0] <= "\u9fff":
            terms.update(word[i:i + 2] for i in range(len(word) - 1))
        elif word not in _STOP:
            terms.add(word)
    return terms


def _navigation(line: str) -> bool:
    if not _LINK.search(line):
        return False
    residue = _LINK.sub("", line)
    return not residue.strip(" \t\r\n*+-|.0123456789[]()!#>")


def select_page_content(raw: str, query: str, *, budget: int = 12000,
                        scan_limit: int = 300000) -> PageContent:
    """跳过链接导航，按词/汉字二元组覆盖排序，优先保留命中段及相邻段。

    片段按原文顺序输出；spans是原始字符串的半开区间。片段之间仅加空行，
    不能假定跨片段拼接的句子属于原文；引用方必须再检查quote在raw中存在。
    无命中时保留清理后的开头，不作无证据的事实补全。
    """
    if budget < 1 or scan_limit < 1:
        raise ValueError("invalid page selection budget")
    content = raw[:scan_limit]
    blocks = []
    offset = navigation_chars = 0
    in_code = False
    for line in content.splitlines(keepends=True):
        if line.lstrip().startswith(("```", "~~~")):
            in_code = not in_code
        if not in_code and _navigation(line):
            navigation_chars += len(line)
        elif line.strip():
            for start in range(offset, offset + len(line), 1000):
                blocks.append((start, min(start + 1000, offset + len(line))))
        offset += len(line)
    terms = _terms(query)
    # 排序只看可见文字，避免URL/标题属性中的同名词造成虚假相关性。
    visible = [_LINK.sub(lambda m: m[0].split("](", 1)[0].lstrip("!["),
                         content[start:end]) for start, end in blocks]
    term_sets = [_terms(text) for text in visible]
    frequencies = {t: sum(t in ts for ts in term_sets) for t in terms}
    weights = {t: 1 + math.log((len(blocks) + 1) / (frequencies[t] + 1)) for t in terms}
    # 保留最早的实质正文（引言/信息表），避免同名后代和重复参考文献挤掉生平首段。
    lead = []
    lead_size = 0
    for i, text in enumerate(visible):
        if len(text.strip()) >= 40 and lead_size < budget // 3:
            lead.append(i)
            lead_size += blocks[i][1] - blocks[i][0]
    remaining_indices = set(range(len(blocks))) - set(lead)
    ranked = []
    covered = set().union(*(term_sets[i] & terms for i in lead)) if lead else set()
    while remaining_indices:
        def score(i: int) -> tuple[float, int]:
            hits = term_sets[i] & terms
            # 新查询词的边际覆盖优先，降低重复主体名称的收益。
            value = sum(weights[t] * (0.15 if t in covered else 1) for t in hits)
            return value / (1 + len(visible[i]) / 1000), -i

        i = max(remaining_indices, key=score)
        if not term_sets[i] & terms:
            break
        ranked.append(i)
        covered.update(term_sets[i] & terms)
        remaining_indices.remove(i)
        if len(ranked) >= 50:  # 有界CPU预算，不为巨大页面重复排序全部块
            break
    neighbors = [j for i in lead + ranked for j in (i - 1, i + 1) if 0 <= j < len(blocks)]
    order = dict.fromkeys(lead + ranked + neighbors + list(range(len(blocks))))
    chosen = []
    size = 0
    for i in order:
        if size >= budget:
            break
        separator = 2 if chosen else 0
        remaining = budget - size - separator
        if remaining <= 0:
            break
        start, end = blocks[i]
        end = min(end, start + remaining)
        chosen.append((start, end))
        size += end - start + separator
    # 合并相邻原文，避免人为割断原有连续段落。
    merged: list[tuple[int, int]] = []
    for start, end in sorted(chosen):
        if (merged and start - merged[-1][1] <= 2
                and not content[merged[-1][1]:start].strip()):
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    pieces = []
    spans = []
    remaining = budget
    for start, end in merged:
        separator = 2 if pieces else 0
        if remaining <= separator:
            break
        end = min(end, start + remaining - separator)
        pieces.append(content[start:end])
        spans.append((start, end))
        remaining -= end - start + separator
    return PageContent("\n\n".join(pieces), spans, len(content), navigation_chars)
