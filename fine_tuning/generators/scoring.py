from __future__ import annotations

from collections.abc import Mapping

from generators.domain_models import (
    CandidateAction,
    CandidateScore,
    CanonicalScenario,
    ScoreContribution,
)
from generators.policy_loader import ScoringPolicy


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _ratio(intersection_size: int, denominator_size: int) -> float:
    if denominator_size == 0:
        return 0.0
    return _clamp(intersection_size / denominator_size)


def _contradiction_pairs(scenario: CanonicalScenario) -> frozenset[tuple[str, str]]:
    evidence_ids = {item.evidence_id for item in scenario.evidence}
    pairs: set[tuple[str, str]] = set()
    for item in scenario.evidence:
        for conflicting_id in item.conflicts_with_evidence_ids:
            if conflicting_id in evidence_ids and conflicting_id != item.evidence_id:
                pairs.add(tuple(sorted((item.evidence_id, conflicting_id))))
    return frozenset(pairs)


def contradiction_count(scenario: CanonicalScenario) -> int:
    """Count unique, explicitly declared evidence contradictions."""
    return len(_contradiction_pairs(scenario))


def _threat_counter_values(
    candidate: CandidateAction,
    scenario: CanonicalScenario,
) -> tuple[float, float]:
    ranked_threats = sorted(
        scenario.threats,
        key=lambda threat: (-threat.priority, threat.entity_id),
    )
    if not ranked_threats:
        return 0.0, 0.0

    primary = ranked_threats[0]
    primary_value = (
        _clamp(primary.priority)
        if primary.entity_id in candidate.countered_threat_ids
        else 0.0
    )
    secondary = ranked_threats[1:]
    if not secondary:
        return primary_value, 0.0
    secondary_value = sum(
        _clamp(threat.priority)
        for threat in secondary
        if threat.entity_id in candidate.countered_threat_ids
    ) / len(secondary)
    return primary_value, _clamp(secondary_value)


def component_values(
    candidate: CandidateAction,
    scenario: CanonicalScenario,
    stale_context_seconds: int,
) -> Mapping[str, float]:
    """Compute declared, normalized inputs for every scoring component."""
    primary_counter, secondary_counter = _threat_counter_values(
        candidate, scenario
    )
    plan = scenario.team_plan
    candidate_evidence = {
        evidence.evidence_id: evidence
        for evidence in scenario.evidence
        if evidence.evidence_id in candidate.evidence_ids
    }
    stale_count = sum(
        evidence.freshness_seconds > stale_context_seconds
        for evidence in candidate_evidence.values()
    )
    evidence_count = len(candidate_evidence)
    contradiction_pairs = _contradiction_pairs(scenario)
    conflicting_ids = {
        evidence_id
        for pair in contradiction_pairs
        for evidence_id in pair
    }
    candidate_conflicts = sum(
        evidence_id in conflicting_ids for evidence_id in candidate_evidence
    )

    return {
        "primaryThreatCounter": primary_counter,
        "secondaryThreatCounter": secondary_counter,
        "supportsWinCondition": _ratio(
            len(candidate.win_condition_tags & plan.win_condition_tags),
            len(plan.win_condition_tags),
        ),
        "fillsMissingFunction": _ratio(
            len(candidate.supports_functions & plan.missing_functions),
            len(plan.missing_functions),
        ),
        "feasibility": _clamp(candidate.feasibility),
        "urgencyAlignment": _clamp(candidate.urgency_alignment),
        "opportunityCost": _clamp(candidate.opportunity_cost),
        "redundancy": _ratio(
            len(candidate.supports_functions & plan.covered_functions),
            len(candidate.supports_functions),
        ),
        "executionBurden": _clamp(candidate.execution_burden),
        "staleEvidencePenalty": _ratio(stale_count, evidence_count),
        "conflictingEvidencePenalty": _ratio(
            candidate_conflicts, evidence_count
        ),
    }


def score_candidate(
    candidate: CandidateAction,
    scenario: CanonicalScenario,
    policy: ScoringPolicy,
    stale_context_seconds: int,
) -> CandidateScore:
    """Score one candidate and retain the exact component decomposition."""
    values = component_values(candidate, scenario, stale_context_seconds)
    contributions: list[ScoreContribution] = []
    for component in policy.components:
        if component.id not in values:
            raise ValueError(f"componente scoring non implementato: {component.id}")
        raw_value = values[component.id]
        sign = 1.0 if component.direction == "REWARD" else -1.0
        contribution = sign * component.weight * raw_value
        contributions.append(
            ScoreContribution(
                component_id=component.id,
                raw_value=raw_value,
                direction=component.direction,
                weight=component.weight,
                contribution=contribution,
            )
        )
    unclamped_total = sum(item.contribution for item in contributions)
    total = _clamp(
        unclamped_total,
        policy.output_range.minimum,
        policy.output_range.maximum,
    )
    evidence_ids = {evidence.evidence_id for evidence in scenario.evidence}
    candidate_has_known_evidence = bool(
        set(candidate.evidence_ids) & evidence_ids
    )
    return CandidateScore(
        action_id=candidate.action_id,
        contributions=tuple(contributions),
        unclamped_total=unclamped_total,
        total=total,
        valid=candidate.feasibility > 0.0 and candidate_has_known_evidence,
    )


def score_candidates(
    scenario: CanonicalScenario,
    policy: ScoringPolicy,
    stale_context_seconds: int,
) -> tuple[CandidateScore, ...]:
    """Score all candidates without assigning meaning to their identifiers."""
    return tuple(
        score_candidate(
            candidate,
            scenario,
            policy,
            stale_context_seconds,
        )
        for candidate in scenario.candidates
    )
