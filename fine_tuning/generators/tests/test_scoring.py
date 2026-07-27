from __future__ import annotations

import math
from dataclasses import replace

import pytest

from generators.domain_models import (
    CandidateAction,
    CanonicalScenario,
    Evidence,
    ScenarioContext,
    TeamPlan,
    Threat,
)
from generators.policy_loader import load_policies
from generators.scoring import (
    component_values,
    contradiction_count,
    score_candidate,
)


POLICIES = load_policies()
STALE_SECONDS = POLICIES.decision.thresholds.stale_context_seconds


def candidate(**changes: object) -> CandidateAction:
    base = CandidateAction(
        action_id="action_a",
        action_type="PEEL",
        evidence_ids=("ev_a",),
        feasibility=0.9,
        supports_functions=frozenset({"PEEL"}),
        countered_threat_ids=frozenset({"threat_primary"}),
        win_condition_tags=frozenset({"PROTECT"}),
        urgency_alignment=0.8,
        opportunity_cost=0.1,
        execution_burden=0.1,
        equivalence_key="protect",
    )
    return replace(base, **changes)


def scenario(
    selected_candidate: CandidateAction | None = None,
    **changes: object,
) -> CanonicalScenario:
    base = CanonicalScenario(
        scenario_id="scenario",
        family_id="family",
        split_group="group",
        source_type="SYNTHETIC",
        seed=7,
        task="THREAT_ASSESSMENT",
        context=ScenarioContext(
            observed_at_game_second=100,
            freshness_seconds=2,
            completeness=0.95,
            required_fields=frozenset({"threats", "plan"}),
            available_fields=frozenset({"threats", "plan"}),
            state_signature="state_a",
        ),
        evidence=(
            Evidence(
                evidence_id="ev_a",
                category="DERIVED_RUNTIME",
                confidence=0.95,
                freshness_seconds=2,
            ),
        ),
        candidates=(selected_candidate or candidate(),),
        threats=(
            Threat("threat_primary", 0.9, ("ev_a",)),
            Threat("threat_secondary", 0.6, ("ev_a",)),
        ),
        team_plan=TeamPlan(
            primary_win_condition="PROTECT_CARRY",
            win_condition_tags=frozenset({"PROTECT"}),
            missing_functions=frozenset({"PEEL", "VISION_CONTROL"}),
            covered_functions=frozenset({"FRONTLINE"}),
        ),
    )
    return replace(base, **changes)


def values_for(
    selected_candidate: CandidateAction | None = None,
    current_scenario: CanonicalScenario | None = None,
) -> dict[str, float]:
    action = selected_candidate or candidate()
    result = component_values(
        action,
        current_scenario or scenario(action),
        STALE_SECONDS,
    )
    return dict(result)


def test_score_contains_every_policy_component() -> None:
    result = score_candidate(
        candidate(), scenario(), POLICIES.scoring, STALE_SECONDS
    )

    assert {item.component_id for item in result.contributions} == {
        component.id for component in POLICIES.scoring.components
    }


def test_score_total_is_exact_contribution_sum() -> None:
    result = score_candidate(
        candidate(), scenario(), POLICIES.scoring, STALE_SECONDS
    )

    assert result.unclamped_total == pytest.approx(
        sum(item.contribution for item in result.contributions)
    )


def test_action_id_does_not_change_component_values() -> None:
    original = candidate()
    renamed = replace(original, action_id="opaque_other")

    assert values_for(original) == values_for(
        renamed, scenario(renamed)
    )


def test_primary_threat_uses_highest_priority_declared_threat() -> None:
    result = values_for()

    assert result["primaryThreatCounter"] == pytest.approx(0.9)


def test_secondary_threats_are_aggregated() -> None:
    action = candidate(
        countered_threat_ids=frozenset(
            {"threat_primary", "threat_secondary", "threat_third"}
        )
    )
    current = scenario(
        action,
        threats=(
            Threat("threat_primary", 0.9),
            Threat("threat_secondary", 0.6),
            Threat("threat_third", 0.3),
        ),
    )

    assert values_for(action, current)["secondaryThreatCounter"] == pytest.approx(
        0.45
    )


def test_multiple_missing_functions_use_coverage_ratio() -> None:
    action = candidate(supports_functions=frozenset({"PEEL"}))

    assert values_for(action)["fillsMissingFunction"] == pytest.approx(0.5)


def test_win_condition_requires_explicit_tag_overlap() -> None:
    aligned = values_for()["supportsWinCondition"]
    unaligned = values_for(
        candidate(win_condition_tags=frozenset({"OTHER"}))
    )["supportsWinCondition"]

    assert aligned == 1.0
    assert unaligned == 0.0


def test_redundancy_penalizes_covered_functions() -> None:
    action = candidate(supports_functions=frozenset({"FRONTLINE"}))
    result = score_candidate(action, scenario(action), POLICIES.scoring, STALE_SECONDS)
    redundancy = next(
        item for item in result.contributions if item.component_id == "redundancy"
    )

    assert redundancy.raw_value == 1.0
    assert redundancy.contribution < 0


def test_opportunity_cost_is_negative() -> None:
    result = score_candidate(
        candidate(opportunity_cost=1.0),
        scenario(candidate(opportunity_cost=1.0)),
        POLICIES.scoring,
        STALE_SECONDS,
    )
    contribution = next(
        item
        for item in result.contributions
        if item.component_id == "opportunityCost"
    )

    assert contribution.contribution == pytest.approx(-contribution.weight)


def test_execution_burden_is_negative() -> None:
    action = candidate(execution_burden=1.0)
    result = score_candidate(action, scenario(action), POLICIES.scoring, STALE_SECONDS)
    contribution = next(
        item
        for item in result.contributions
        if item.component_id == "executionBurden"
    )

    assert contribution.contribution < 0


def test_lower_feasibility_lowers_score() -> None:
    high = candidate(feasibility=1.0)
    low = candidate(feasibility=0.1)

    high_score = score_candidate(
        high, scenario(high), POLICIES.scoring, STALE_SECONDS
    ).total
    low_score = score_candidate(
        low, scenario(low), POLICIES.scoring, STALE_SECONDS
    ).total
    assert low_score < high_score


def test_stale_candidate_evidence_adds_penalty() -> None:
    current = scenario(
        evidence=(
            Evidence(
                "ev_a",
                "DERIVED_RUNTIME",
                0.95,
                STALE_SECONDS + 1,
            ),
        )
    )

    assert values_for(current_scenario=current)["staleEvidencePenalty"] == 1.0


def test_conflicting_candidate_evidence_adds_penalty() -> None:
    current = scenario(
        evidence=(
            Evidence(
                "ev_a",
                "DERIVED_RUNTIME",
                0.95,
                2,
                conflicts_with_evidence_ids=frozenset({"ev_b"}),
            ),
            Evidence(
                "ev_b",
                "DERIVED_RUNTIME",
                0.95,
                2,
                conflicts_with_evidence_ids=frozenset({"ev_a"}),
            ),
        )
    )

    assert contradiction_count(current) == 1
    assert values_for(current_scenario=current)[
        "conflictingEvidencePenalty"
    ] == 1.0


def test_no_threats_produce_zero_threat_components() -> None:
    current = scenario(threats=())
    result = values_for(current_scenario=current)

    assert result["primaryThreatCounter"] == 0.0
    assert result["secondaryThreatCounter"] == 0.0


def test_candidate_without_known_evidence_is_invalid() -> None:
    action = candidate(evidence_ids=("unknown",))
    result = score_candidate(
        action, scenario(action), POLICIES.scoring, STALE_SECONDS
    )

    assert result.valid is False


def test_zero_feasibility_candidate_is_invalid() -> None:
    action = candidate(feasibility=0.0)
    result = score_candidate(
        action, scenario(action), POLICIES.scoring, STALE_SECONDS
    )

    assert result.valid is False
