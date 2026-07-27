from __future__ import annotations

from dataclasses import dataclass

from generators.domain_models import (
    CanonicalScenario,
    OracleResult,
)


EXPECTED_EFFECTS = frozenset(
    {
        "SAME_DECISION",
        "CHANGE_ACTION",
        "CHANGE_TO_SUPPRESS",
        "CHANGE_TO_REFRESH",
        "CHANGE_FROM_REFRESH",
    }
)


@dataclass(frozen=True, slots=True)
class CounterfactualPair:
    rule_id: str
    expected_effect: str
    observed_effect: str
    parent: CanonicalScenario
    counterfactual: CanonicalScenario
    parent_result: OracleResult
    counterfactual_result: OracleResult


def classify_effect(
    parent_result: OracleResult,
    child_result: OracleResult,
) -> str:
    parent = parent_result.decision
    child = child_result.decision
    if (
        parent.decision == child.decision
        and parent.primary_action_id == child.primary_action_id
    ):
        return "SAME_DECISION"
    if child.decision == "REQUEST_REFRESH":
        return "CHANGE_TO_REFRESH"
    if parent.decision == "REQUEST_REFRESH":
        return "CHANGE_FROM_REFRESH"
    if child.decision == "SUPPRESS":
        return "CHANGE_TO_SUPPRESS"
    return "CHANGE_ACTION"
