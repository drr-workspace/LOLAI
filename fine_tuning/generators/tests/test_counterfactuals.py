from __future__ import annotations

from dataclasses import replace

import pytest

from generators.counterfactuals.apply import (
    apply_counterfactual,
    load_rules,
)
from generators.domain_models import (
    CandidateAction,
    CanonicalScenario,
    Evidence,
    RecentAdvice,
    ScenarioContext,
    TeamPlan,
    Threat,
)


RULE_IDS = tuple(load_rules())
SEEDS = (101, 202, 303, 404)
DIMENSION_BY_RULE = {
    "SWAP_PRIMARY_SECONDARY_THREAT": "threats",
    "REDUCE_EVIDENCE_FRESHNESS": "evidence",
    "INTRODUCE_CONTRADICTION": "evidence",
    "REMOVE_REQUIRED_EVIDENCE": "evidence",
    "MAKE_WINNER_INFEASIBLE": "candidates",
    "MAKE_LOSER_IMMEDIATELY_FEASIBLE": "candidates",
    "FILL_MISSING_TEAM_FUNCTION": "team_plan",
    "MAKE_CANDIDATES_EQUIVALENT": "candidates",
    "ADD_MEANINGFUL_STATE_CHANGE": "context",
    "REMOVE_MEANINGFUL_STATE_CHANGE": "context",
    "INCREASE_OBJECTIVE_URGENCY": "candidates",
    "RESOLVE_VISION_GAP": "evidence",
    "CHANGE_WAVE_STATE": "context",
    "AGE_RECENT_ADVICE": "recent_advice",
    "ADD_NEW_RELIABLE_EVIDENCE": "evidence",
}


def _base_scenario(seed: int) -> CanonicalScenario:
    evidence = tuple(
        Evidence(f"evidence_{index}", "OBSERVED_RUNTIME", 0.95, 1)
        for index in range(3)
    )
    threats = (
        Threat("entity_primary", 0.95, ("evidence_0",)),
        Threat("entity_secondary", 0.2, ("evidence_1",)),
    )
    shared = {
        "action_type": "PEEL",
        "evidence_ids": ("evidence_0",),
        "feasibility": 0.9,
        "supports_functions": frozenset({"PEEL"}),
        "win_condition_tags": frozenset({"PROTECT"}),
        "urgency_alignment": 0.7,
        "opportunity_cost": 0.1,
        "execution_burden": 0.1,
    }
    candidates = (
        CandidateAction(
            "action_primary",
            countered_threat_ids=frozenset({"entity_primary"}),
            equivalence_key="primary_response",
            **shared,
        ),
        CandidateAction(
            "action_secondary",
            countered_threat_ids=frozenset({"entity_secondary"}),
            equivalence_key="secondary_response",
            **shared,
        ),
    )
    return CanonicalScenario(
        scenario_id=f"parent_{seed}",
        family_id="counterfactual_test",
        split_group="counterfactual_test",
        source_type="SYNTHETIC",
        seed=seed,
        task="THREAT_ASSESSMENT",
        context=ScenarioContext(
            observed_at_game_second=600,
            freshness_seconds=1,
            completeness=0.95,
            required_fields=frozenset({"threatState"}),
            available_fields=frozenset({"threatState"}),
            state_signature="state_same",
        ),
        evidence=evidence,
        candidates=candidates,
        threats=threats,
        team_plan=TeamPlan(
            primary_win_condition="PROTECT_PLAN",
            win_condition_tags=frozenset({"PROTECT"}),
            missing_functions=frozenset({"PEEL"}),
            covered_functions=frozenset({"FRONTLINE"}),
        ),
    )


def _scenario_for(rule_id: str, seed: int) -> CanonicalScenario:
    scenario = _base_scenario(seed)
    if rule_id == "MAKE_LOSER_IMMEDIATELY_FEASIBLE":
        first = replace(
            scenario.candidates[0],
            countered_threat_ids=frozenset(),
            urgency_alignment=0.5,
        )
        second = replace(
            scenario.candidates[1],
            countered_threat_ids=frozenset(),
            feasibility=0.01,
            urgency_alignment=0.5,
        )
        return replace(scenario, candidates=(first, second))
    if rule_id in {"ADD_MEANINGFUL_STATE_CHANGE", "AGE_RECENT_ADVICE"}:
        return replace(
            scenario,
            recent_advice=(
                RecentAdvice(
                    "action_primary",
                    "primary_response",
                    5,
                    "SHOW",
                    "state_same",
                ),
            ),
        )
    if rule_id == "REMOVE_MEANINGFUL_STATE_CHANGE":
        return replace(
            scenario,
            context=replace(
                scenario.context, state_signature="state_changed"
            ),
            recent_advice=(
                RecentAdvice(
                    "action_primary",
                    "primary_response",
                    5,
                    "SHOW",
                    "state_same",
                ),
            ),
        )
    if rule_id == "ADD_NEW_RELIABLE_EVIDENCE":
        return replace(
            scenario,
            evidence=tuple(
                replace(item, confidence=0.1)
                for item in scenario.evidence
            ),
        )
    return scenario


@pytest.mark.parametrize("rule_id", RULE_IDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_counterfactual_effects_match_declaration(
    rule_id: str, seed: int
) -> None:
    pair = apply_counterfactual(_scenario_for(rule_id, seed), rule_id)

    assert pair.observed_effect == pair.expected_effect
    assert pair.counterfactual.parent_scenario_id == pair.parent.scenario_id
    assert pair.counterfactual.counterfactual_pair_id
    assert pair.counterfactual.source_type == "COUNTERFACTUAL"


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_counterfactual_changes_one_primary_dimension(rule_id: str) -> None:
    pair = apply_counterfactual(_scenario_for(rule_id, 505), rule_id)
    semantic_fields = (
        "context",
        "evidence",
        "threats",
        "candidates",
        "team_plan",
        "recent_advice",
    )
    changed = {
        field
        for field in semantic_fields
        if getattr(pair.parent, field)
        != getattr(pair.counterfactual, field)
    }

    assert changed == {DIMENSION_BY_RULE[rule_id]}


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_counterfactual_is_deterministic(rule_id: str) -> None:
    scenario = _scenario_for(rule_id, 606)

    assert apply_counterfactual(
        scenario, rule_id
    ) == apply_counterfactual(scenario, rule_id)
