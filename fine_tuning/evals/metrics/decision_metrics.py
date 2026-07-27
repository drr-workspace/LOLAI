from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def compute(samples: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    total = len(samples)
    decision = primary = acceptable = priority = 0
    true_reasons = predicted_reasons = matched_reasons = 0
    for sample in samples:
        expected = sample["expected"]
        predicted = sample.get("prediction") or {}
        decision += int(predicted.get("decision") == expected.get("decision"))
        primary += int(
            predicted.get("primaryActionId") == expected.get("primaryActionId")
        )
        acceptable_ids = {
            expected.get("primaryActionId"),
            *expected.get("alternativeActionIds", []),
        } - {None}
        predicted_id = predicted.get("primaryActionId")
        acceptable += int(
            predicted_id in acceptable_ids
            or (
                expected.get("decision") != "SHOW"
                and predicted_id is None
            )
        )
        priority += int(predicted.get("priority") == expected.get("priority"))
        truth = set(expected.get("reasonCodes", []))
        guess = set(predicted.get("reasonCodes", []))
        true_reasons += len(truth)
        predicted_reasons += len(guess)
        matched_reasons += len(truth & guess)
    precision = _safe_ratio(matched_reasons, predicted_reasons)
    recall = _safe_ratio(matched_reasons, true_reasons)
    f1 = _safe_ratio(2 * precision * recall, precision + recall)
    return {
        "decisionAccuracy": _safe_ratio(decision, total),
        "primaryActionIdAccuracy": _safe_ratio(primary, total),
        "acceptableActionRate": _safe_ratio(acceptable, total),
        "priorityAccuracy": _safe_ratio(priority, total),
        "reasonCodePrecision": precision,
        "reasonCodeRecall": recall,
        "reasonCodeF1": f1,
    }
