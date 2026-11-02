import pytest

from lychee_mas.methods.build.mindsearch.answer_format import format_answer


@pytest.mark.parametrize("text,expected", [
    ("Answer: Paris\nbecause ...", "Paris"),
    ("**Answer:** 巴黎。\n因为 ...", "巴黎"),
    ("answer：yes", "yes"),
    ("Answer: 2001.\nreleased in 2001", "2001"),
])
def test_format_ok(text, expected):
    out = format_answer(text)
    assert out.short == expected and out.format_ok is True


def test_fallback_marks_bad_format():
    out = format_answer("the answer is Paris")
    assert out.format_ok is False and out.short


def test_empty():
    assert format_answer("").short == ""