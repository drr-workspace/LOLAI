from __future__ import annotations

from dataclasses import fields, replace

import pytest

from generators.domain_models import (
    CandidateAction,
    CanonicalScenario,
    Evidence,
    RecentAdvice,
    ScenarioContext,
    TeamPlan,
    Threat,
)
from generators.ontology_registry import load_ontology
from generators.oracle import StrategicOracle
from generators.oracle_trace import (
    reconstruct_candidate_total,
    reconstruct_confidence,
    trace_is_consistent,
)


ORACLE = StrategicOracle()
ONTOLOGY = load_ontology()


def good_candidate(action_id: str = "action_good", **changes: object) -> CandidateAction:
    base = CandidateAction(
        action_id=action_id,
        action_type="PEEL",
        evidence_ids=("ev_good",),
        feasibility=0.95,
        supports_functions=frozenset({"PEEL"}),
        countered_threat_ids=frozenset({"enemy_primary", "enemy_secondary"}),
        win_condition_tags=frozenset({"PROTECT"}),
        urgency_alignment=0.9,
        opportunity_cost=0.05,
        execution_burden=0.05,
        equivalence_key="protect_primary",
    )
    return replace(base, **changes)


def weak_candidate(action_id: str = "action_weak", **changes: object) -> CandidateAction:
    base = CandidateAction(
        action_id=action_id,
        action_type="ENGAGE",
        evidence_ids=("ev_weak",),
        feasibility=0.7,
        supports_functions=frozenset({"FRONTLINE"}),
        urgency_alignment=0.2,
        opportunity_cost=0.7,
        execution_burden=0.7,
        equivalence_key="engage",
    )
    return replace(base, **changes)


def base_scenario(**changes: object) -> CanonicalScenario:
    base = CanonicalScenario(
        scenario_id="scenario_a",
        family_id="family_a",
        split_group="group_a",
        source_type="SYNTHETIC",
        seed=101,
        task="THREAT_ASSESSMENT",
        context=ScenarioContext(
            observed_at_game_second=500,
            freshness_seconds=2,
            completeness=0.95,
            uncertain_fields=(),
            required_fields=frozenset({"threats", "teamPlan"}),
            available_fields=frozenset({"threats", "teamPlan"}),
            state_signature="state_1",
        ),
        evidence=(
            Evidence("ev_good", "DERIVED_RUNTIME", 0.95, 2),
            Evidence("ev_weak", "STRATEGIC_KNOWLEDGE", 0.9, 1),
        ),
        candidates=(good_candidate(), weak_candidate()),
        threats=(
            Threat("enemy_primary", 0.95, ("ev_good",)),
            Threat("enemy_secondary", 0.6, ("ev_good",)),
        ),
        team_plan=TeamPlan(
            primary_win_condition="PROTECT_CARRY",
            win_condition_tags=frozenset({"PROTECT"}),
            missing_functions=frozenset({"PEEL"}),
            covered_functions=frozenset({"FRONTLINE"}),
        ),
    )
    return replace(base, **changes)


def test_candidate_order_does_not_change_decision() -> None:
    original = base_scenario()
    reversed_scenario = replace(original, candidates=tuple(reversed(original.candidates)))

    first = ORACLE.decide(original)
    second = ORACLE.decide(reversed_scenario)

    assert first.decision == second.decision
    assert first.decision.primary_action_id == second.decision.primary_action_id


def test_changing_action_ids_does_not_change_semantic_decision() -> None:
    original = base_scenario()
    renamed = replace(
        original,
        candidates=(
            replace(original.candidates[0], action_id="opaque_x"),
            replace(original.candidates[1], action_id="opaque_y"),
        ),
    )

    first = ORACLE.decide(original)
    second = ORACLE.decide(renamed)

    assert first.decision.decision == second.decision.decision == "SHOW"
    assert first.decision.confidence == second.decision.confidence
    assert second.decision.primary_action_id == "opaque_x"


def test_selected_candidate_always_exists_in_scenario() -> None:
    current = base_scenario()
    result = ORACLE.decide(current)

    assert result.decision.primary_action_id in {
        candidate.action_id for candidate in current.candidates
    }
    assert result.decision.primary_action_id != "not_present"


def test_stale_context_causes_request_refresh() -> None:
    current = base_scenario(
        context=replace(base_scenario().context, freshness_seconds=31)
    )

    result = ORACLE.decide(current)

    assert result.decision.decision == "REQUEST_REFRESH"
    assert result.trace.triggered_gate == "staleRequiredContext"


def test_stale_evidence_causes_request_refresh() -> None:
    current = base_scenario(
        evidence=(
            Evidence("ev_good", "DERIVED_RUNTIME", 0.95, 31),
            Evidence("ev_weak", "STRATEGIC_KNOWLEDGE", 0.9, 1),
        )
    )

    assert ORACLE.decide(current).decision.decision == "REQUEST_REFRESH"


def test_incomplete_context_causes_request_refresh() -> None:
    current = base_scenario(
        context=replace(base_scenario().context, completeness=0.5)
    )

    assert ORACLE.decide(current).trace.triggered_gate == "incompleteContext"


def test_missing_required_field_causes_request_refresh() -> None:
    current = base_scenario(
        context=replace(
            base_scenario().context,
            available_fields=frozenset({"threats"}),
        )
    )

    result = ORACLE.decide(current)

    assert result.decision.decision == "REQUEST_REFRESH"
    assert result.trace.missing_required_fields == ("teamPlan",)


def test_unreliable_evidence_causes_request_refresh() -> None:
    current = base_scenario(
        evidence=(
            Evidence("ev_good", "DERIVED_RUNTIME", 0.2, 2),
            Evidence("ev_weak", "STRATEGIC_KNOWLEDGE", 0.2, 1),
        )
    )

    assert ORACLE.decide(current).trace.triggered_gate == "unreliableEvidence"


def test_excessive_contradictions_cause_request_refresh() -> None:
    current = base_scenario(
        evidence=(
            Evidence(
                "ev_good",
                "DERIVED_RUNTIME",
                0.95,
                2,
                conflicts_with_evidence_ids=frozenset({"ev_weak"}),
            ),
            Evidence(
                "ev_weak",
                "STRATEGIC_KNOWLEDGE",
                0.9,
                1,
                conflicts_with_evidence_ids=frozenset({"ev_good", "ev_third"}),
            ),
            Evidence(
                "ev_third",
                "OBSERVED_RUNTIME",
                0.9,
                1,
                conflicts_with_evidence_ids=frozenset({"ev_weak"}),
            ),
        )
    )

    result = ORACLE.decide(current)

    assert result.decision.decision == "REQUEST_REFRESH"
    assert result.trace.contradiction_count == 2


def test_equivalent_candidates_cause_suppress() -> None:
    first = good_candidate("first")
    second = good_candidate("second")
    current = base_scenario(candidates=(first, second))

    result = ORACLE.decide(current)

    assert result.decision.decision == "SUPPRESS"
    assert result.trace.triggered_gate == "insufficientScoreMargin"


def test_feasibility_counterfactual_can_change_selected_candidate() -> None:
    first = good_candidate("first")
    second = good_candidate(
        "second",
        countered_threat_ids=frozenset({"enemy_primary"}),
        urgency_alignment=0.1,
        execution_burden=0.4,
    )
    baseline = base_scenario(candidates=(first, second))
    counterfactual = replace(
        baseline,
        candidates=(replace(first, feasibility=0.0), second),
        source_type="COUNTERFACTUAL",
    )

    baseline_result = ORACLE.decide(baseline)
    counterfactual_result = ORACLE.decide(counterfactual)

    assert baseline_result.decision.primary_action_id == "first"
    assert counterfactual_result.decision.primary_action_id == "second"


def test_recent_advice_without_state_change_causes_suppress() -> None:
    current = base_scenario(
        recent_advice=(
            RecentAdvice(
                action_id="old_id",
                equivalence_key="protect_primary",
                age_seconds=5,
                state_signature="state_1",
            ),
        )
    )

    result = ORACLE.decide(current)

    assert result.decision.decision == "SUPPRESS"
    assert result.trace.triggered_gate == "recentEquivalentAdvice"


def test_significant_state_change_allows_new_advice() -> None:
    current = base_scenario(
        recent_advice=(
            RecentAdvice(
                action_id="action_good",
                equivalence_key="protect_primary",
                age_seconds=5,
                state_signature="old_state",
            ),
        )
    )

    assert ORACLE.decide(current).decision.decision == "SHOW"


def test_expired_recent_advice_does_not_suppress() -> None:
    current = base_scenario(
        recent_advice=(
            RecentAdvice(
                action_id="action_good",
                equivalence_key="protect_primary",
                age_seconds=21,
                state_signature="state_1",
            ),
        )
    )

    assert ORACLE.decide(current).decision.decision == "SHOW"


def test_non_show_recent_record_does_not_suppress() -> None:
    current = base_scenario(
        recent_advice=(
            RecentAdvice(
                action_id="action_good",
                equivalence_key="protect_primary",
                age_seconds=5,
                decision="SUPPRESS",
                state_signature="state_1",
            ),
        )
    )

    assert ORACLE.decide(current).decision.decision == "SHOW"


def test_no_candidates_causes_suppress() -> None:
    result = ORACLE.decide(base_scenario(candidates=()))

    assert result.decision.decision == "SUPPRESS"
    assert result.trace.triggered_gate == "noValidCandidate"


def test_all_non_feasible_candidates_cause_suppress() -> None:
    current = base_scenario(
        candidates=(
            good_candidate(feasibility=0.0),
            weak_candidate(feasibility=0.0),
        )
    )

    assert ORACLE.decide(current).trace.triggered_gate == "noValidCandidate"


def test_low_value_candidate_causes_suppress() -> None:
    low = weak_candidate(
        feasibility=0.05,
        opportunity_cost=1.0,
        execution_burden=1.0,
    )
    current = base_scenario(candidates=(low,))

    result = ORACLE.decide(current)

    assert result.decision.decision == "SUPPRESS"
    assert result.trace.triggered_gate == "lowActionValue"


def test_single_strong_candidate_can_show_without_second_candidate() -> None:
    result = ORACLE.decide(base_scenario(candidates=(good_candidate(),)))

    assert result.decision.decision == "SHOW"
    assert result.decision.primary_action_id == "action_good"


def test_show_alternatives_are_present_candidates() -> None:
    current = base_scenario()
    result = ORACLE.decide(current)
    candidate_ids = {candidate.action_id for candidate in current.candidates}

    assert set(result.decision.alternative_action_ids) <= candidate_ids
    assert result.decision.primary_action_id not in result.decision.alternative_action_ids


def test_trace_reconstructs_exact_final_score() -> None:
    result = ORACLE.decide(base_scenario())
    selected = result.decision.primary_action_id
    assert selected is not None
    selected_score = next(
        score
        for score in result.trace.candidate_scores
        if score.action_id == selected
    )

    assert reconstruct_candidate_total(result.trace, selected) == pytest.approx(
        selected_score.unclamped_total
    )
    assert trace_is_consistent(result.trace)


def test_trace_reconstructs_exact_confidence() -> None:
    result = ORACLE.decide(base_scenario())

    assert reconstruct_confidence(result.trace) == pytest.approx(
        result.trace.confidence_unrounded
    )


def test_same_seed_and_scenario_are_deterministic() -> None:
    current = base_scenario(seed=999)

    assert ORACLE.decide(current) == ORACLE.decide(current)


def test_different_seed_does_not_introduce_random_labeling() -> None:
    first = ORACLE.decide(base_scenario(seed=1))
    second = ORACLE.decide(base_scenario(seed=2))

    assert first.decision == second.decision


def test_reason_codes_are_valid_and_compatible() -> None:
    current = base_scenario()
    decision = ORACLE.decide(current).decision

    for reason in decision.reason_codes:
        entry = ONTOLOGY.require("reason-codes", reason)
        assert current.task in entry["allowedTasks"]
        assert decision.decision in entry["decisionCompatibility"]


def test_evidence_ids_are_declared_in_scenario() -> None:
    current = base_scenario()
    decision = ORACLE.decide(current).decision

    assert set(decision.evidence_ids) <= {
        evidence.evidence_id for evidence in current.evidence
    }


def test_primary_and_secondary_threats_affect_ranking() -> None:
    primary_only = good_candidate(
        "primary_only",
        countered_threat_ids=frozenset({"enemy_primary"}),
    )
    both = good_candidate(
        "both",
        countered_threat_ids=frozenset({"enemy_primary", "enemy_secondary"}),
    )
    current = base_scenario(candidates=(primary_only, both))

    result = ORACLE.decide(current)

    assert result.trace.ranked_action_ids[0] == "both"
    assert result.decision.decision == "SUPPRESS"


def test_covering_already_covered_function_is_penalized() -> None:
    redundant = good_candidate(
        "redundant",
        supports_functions=frozenset({"FRONTLINE"}),
    )
    missing = good_candidate("missing", supports_functions=frozenset({"PEEL"}))
    current = base_scenario(candidates=(redundant, missing))

    assert ORACLE.decide(current).decision.primary_action_id == "missing"


def test_win_condition_alignment_affects_ranking() -> None:
    aligned = good_candidate("aligned")
    unaligned = good_candidate(
        "unaligned", win_condition_tags=frozenset({"OTHER"})
    )
    current = base_scenario(candidates=(aligned, unaligned))

    assert ORACLE.decide(current).decision.primary_action_id == "aligned"


def test_uncertain_fields_reduce_confidence() -> None:
    certain = ORACLE.decide(base_scenario()).decision.confidence
    uncertain_context = replace(
        base_scenario().context,
        uncertain_fields=("threats", "teamPlan"),
    )
    uncertain = ORACLE.decide(
        base_scenario(context=uncertain_context)
    ).decision.confidence

    assert uncertain < certain


def test_trace_is_not_part_of_oracle_decision() -> None:
    result = ORACLE.decide(base_scenario())

    assert "trace" not in {field.name for field in fields(result.decision)}


def test_unknown_task_is_rejected() -> None:
    with pytest.raises(ValueError, match="task non valido"):
        ORACLE.decide(base_scenario(task="NOT_A_TASK"))


def test_duplicate_candidate_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="univoci"):
        ORACLE.decide(
            base_scenario(
                candidates=(good_candidate("same"), weak_candidate("same"))
            )
        )
