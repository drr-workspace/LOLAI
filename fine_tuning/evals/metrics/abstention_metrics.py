from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _class_metrics(
    samples: Sequence[Mapping[str, Any]], label: str
) -> dict[str, float]:
    tp = fp = fn = 0
    for sample in samples:
        truth = sample["expected"].get("decision") == label
        guess = (sample.get("prediction") or {}).get("decision") == label
        tp += int(truth and guess)
        fp += int(not truth and guess)
        fn += int(truth and not guess)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
    }


def compute(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    suppress = _class_metrics(samples, "SUPPRESS")
    refresh = _class_metrics(samples, "REQUEST_REFRESH")
    abstain_truth = [
        sample
        for sample in samples
        if sample["expected"].get("decision") != "SHOW"
    ]
    show_truth = [
        sample
        for sample in samples
        if sample["expected"].get("decision") == "SHOW"
    ]
    false_show = sum(
        (sample.get("prediction") or {}).get("decision") == "SHOW"
        for sample in abstain_truth
    )
    unnecessary = sum(
        (sample.get("prediction") or {}).get("decision") == "SUPPRESS"
        for sample in show_truth
    )
    return {
        "suppress": suppress,
        "requestRefresh": refresh,
        "falseShowRate": (
            false_show / len(abstain_truth) if abstain_truth else 0.0
        ),
        "unnecessaryAdviceRate": (
            unnecessary / len(show_truth) if show_truth else 0.0
        ),
    }
