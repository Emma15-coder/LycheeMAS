"""MindSearch问答评分扩展；保留框架exact，显式采用规范化EM。"""
from __future__ import annotations

from lychee_mas.eval import metrics


def normalized_em(prediction: str, gold: str) -> float:
    """复用框架归一化后完全相等；空标准答案不计正确，不做数字/子串匹配。"""
    expected = metrics._norm(gold)
    return float(bool(expected) and metrics._norm(prediction) == expected)


def metric_name(kind: str) -> str:
    return "normalized_em" if kind == "exact" else kind


def score_answer(kind: str, prediction: str, gold) -> float:
    if kind == "exact":
        return normalized_em(prediction, gold)
    return metrics.score(kind, prediction, gold)


def aggregate_samples(samples: list[dict], run_info: dict) -> dict:
    """复用框架二值统计；kind保持任务类别，实际评分指标单独标明。"""
    result = metrics.aggregate_samples(samples, run_info)
    name = metric_name(run_info["scorer_kind"])
    result["score_metric"] = name
    result["quality_metric"] = name
    return result
