from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def _effect(first: Mapping[str, Any], second: Mapping[str, Any]) -> str:
    if (
        first.get("decision") == second.get("decision")
        and first.get("primaryActionId") == second.get("primaryActionId")
    ):
        return "SAME_DECISION"
    if second.get("decision") == "REQUEST_REFRESH":
        return "CHANGE_TO_REFRESH"
    if first.get("decision") == "REQUEST_REFRESH":
        return "CHANGE_FROM_REFRESH"
    if second.get("decision") == "SUPPRESS":
        return "CHANGE_TO_SUPPRESS"
    return "CHANGE_ACTION"


def compute(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, float | int | None]:
    groups: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        pair_id = sample.get("metadata", {}).get("counterfactualPairId")
        if pair_id:
            groups[str(pair_id)].append(sample)
    complete = [items for items in groups.values() if len(items) == 2]
    consistent = expected_change = invariant = invariant_total = 0
    for pair in complete:
        pair = sorted(
            pair,
            key=lambda item: bool(
                item.get("metadata", {}).get("isCounterfactual")
            ),
        )
        expected_effect = pair[1].get("metadata", {}).get("expectedEffect")
        predicted_effect = _effect(
            pair[0].get("prediction") or {},
            pair[1].get("prediction") or {},
        )
        truth_effect = _effect(pair[0]["expected"], pair[1]["expected"])
        consistent += int(predicted_effect == truth_effect)
        expected_change += int(predicted_effect == expected_effect)
        if expected_effect == "SAME_DECISION":
            invariant_total += 1
            invariant += int(predicted_effect == "SAME_DECISION")
    return {
        "completePairCount": len(complete),
        "counterfactualConsistency": (
            consistent / len(complete) if complete else None
        ),
        "expectedChangeAccuracy": (
            expected_change / len(complete) if complete else None
        ),
        "invariantPairConsistency": (
            invariant / invariant_total if invariant_total else None
        ),
        "invariantPairCount": invariant_total,
    }
