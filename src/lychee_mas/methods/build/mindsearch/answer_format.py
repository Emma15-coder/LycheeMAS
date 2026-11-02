"""规划者输出格式化：把 finish 拆成 short answer 与长答案。

为什么放在 build 接缝：答案格式是**本方法的输出契约**，不是评测口径；
eval 只拿到已经成型的短答案，不参与解析。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_ANSWER_LINE = re.compile(
    r"^[ \t>*_#-]*answer[ \t>*_]*[:：][ \t]*(.+?)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_TRAIL = " \t.,;:!?。，；：！？、"


@dataclass(frozen=True)
class FormattedAnswer:
    short: str          # 给打分用的短答案
    long: str           # 完整回答（含解释/来源）
    format_ok: bool     # 是否带规范的 Answer: 行


def _clean(s: str) -> str:
    s = s.strip().strip("*_`").strip("\"'“”‘’")
    return s.strip(_TRAIL)


def format_answer(text: str) -> FormattedAnswer:
    if not text or not text.strip():
        return FormattedAnswer("", "", False)
    m = _ANSWER_LINE.search(text)
    if m:
        return FormattedAnswer(_clean(m.group(1)), text, True)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return FormattedAnswer(_clean(lines[-1]) if lines else "", text, False)