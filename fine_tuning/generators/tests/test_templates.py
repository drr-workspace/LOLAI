from __future__ import annotations

from dataclasses import replace

import pytest

from evals.validators.validate_templates import build_scenario
from generators.oracle import StrategicOracle
from generators.template_loader import (
    TemplateFamily,
    TemplateRepository,
    load_templates,
)


ORACLE = StrategicOracle()
REPOSITORY = load_templates()
NEW_TASKS = ("COMPOSITION_PLAN", "MATCHUP_PLAN")


def _family(task: str) -> TemplateFamily:
    return REPOSITORY.families(task=task)[0]


@pytest.mark.parametrize(
    ("task", "minimum"),
    (("COMPOSITION_PLAN", 20), ("MATCHUP_PLAN", 24)),
)
def test_required_family_counts(task: str, minimum: int) -> None:
    assert len(REPOSITORY.families(task=task)) >= minimum


@pytest.mark.parametrize("task", NEW_TASKS)
def test_each_family_has_declared_causal_structure(task: str) -> None:
    for family in REPOSITORY.families(task=task):
        scenario = build_scenario(REPOSITORY, family)

        assert len(family.causal_parameters) >= 3
        assert len(scenario.candidates) >= 2
        assert family.counterfactual_axes
        assert family.edge_case
        assert family.message_intents


@pytest.mark.parametrize("task", NEW_TASKS)
def test_edge_case_is_correlated_with_a_causal_parameter(task: str) -> None:
    for family in REPOSITORY.families(task=task):
        causal_ids = {
            str(parameter["id"]) for parameter in family.causal_parameters
        }
        edge_scenario = build_scenario(
            REPOSITORY, family, edge_case=True
        )
        result = ORACLE.decide(edge_scenario)

        assert family.edge_case["causalParameterId"] in causal_ids
        assert result.decision.decision in {"SUPPRESS", "REQUEST_REFRESH"}


@pytest.mark.parametrize("task", NEW_TASKS)
def test_candidate_reordering_does_not_change_decision(task: str) -> None:
    scenario = build_scenario(REPOSITORY, _family(task))
    reordered = replace(
        scenario, candidates=tuple(reversed(scenario.candidates))
    )

    original_result = ORACLE.decide(scenario)
    reordered_result = ORACLE.decide(reordered)

    assert original_result.decision == reordered_result.decision


@pytest.mark.parametrize("task", NEW_TASKS)
def test_opaque_candidate_rename_preserves_semantic_decision(task: str) -> None:
    scenario = build_scenario(REPOSITORY, _family(task))
    rename = {
        candidate.action_id: f"opaque_{index}"
        for index, candidate in enumerate(scenario.candidates)
    }
    renamed = replace(
        scenario,
        candidates=tuple(
            replace(candidate, action_id=rename[candidate.action_id])
            for candidate in scenario.candidates
        ),
    )

    original_result = ORACLE.decide(scenario)
    renamed_result = ORACLE.decide(renamed)

    assert original_result.decision.decision == renamed_result.decision.decision
    assert original_result.decision.confidence == renamed_result.decision.confidence
    assert renamed_result.decision.primary_action_id == rename[
        original_result.decision.primary_action_id
    ]


@pytest.mark.parametrize("task", NEW_TASKS)
def test_one_causal_variable_can_change_selected_candidate(task: str) -> None:
    scenario = build_scenario(REPOSITORY, _family(task))
    selected_id = ORACLE.decide(scenario).decision.primary_action_id
    assert selected_id is not None
    changed_candidates = tuple(
        replace(candidate, feasibility=0.0)
        if candidate.action_id == selected_id
        else candidate
        for candidate in scenario.candidates
    )
    counterfactual = replace(
        scenario,
        source_type="COUNTERFACTUAL",
        candidates=changed_candidates,
    )

    changed_result = ORACLE.decide(counterfactual)

    assert changed_result.decision.primary_action_id != selected_id


@pytest.mark.parametrize("task", NEW_TASKS)
def test_dominant_context_variable_changes_decision(task: str) -> None:
    scenario = build_scenario(REPOSITORY, _family(task))
    stale = replace(
        scenario,
        context=replace(scenario.context, freshness_seconds=31),
        source_type="COUNTERFACTUAL",
    )

    assert ORACLE.decide(scenario).decision.decision == "SHOW"
    assert ORACLE.decide(stale).decision.decision == "REQUEST_REFRESH"


@pytest.mark.parametrize("task", NEW_TASKS)
def test_visible_message_intent_is_separate_from_structural_label(
    task: str,
) -> None:
    for family in REPOSITORY.families(task=task):
        assert "messageIntents" not in family.expected_decision_constraints
        assert set(family.message_intents).isdisjoint(
            family.allowed_reason_codes
        )


def test_repository_filters_new_tasks() -> None:
    repository: TemplateRepository = REPOSITORY

    assert all(
        family.task == "COMPOSITION_PLAN"
        for family in repository.families(task="COMPOSITION_PLAN")
    )
    assert all(
        family.task == "MATCHUP_PLAN"
        for family in repository.families(task="MATCHUP_PLAN")
    )


@pytest.mark.parametrize(
    ("task", "minimum"),
    (("MACRO_PRIORITY", 26), ("ADVICE_SUPPRESSION", 14)),
)
def test_batch_3c_family_counts(task: str, minimum: int) -> None:
    assert len(REPOSITORY.families(task=task)) >= minimum
    assert len(REPOSITORY.families()) > 120


@pytest.mark.parametrize(
    "task", ("MACRO_PRIORITY", "ADVICE_SUPPRESSION")
)
def test_batch_3c_semantic_metadata_is_complete(task: str) -> None:
    for family in REPOSITORY.families(task=task):
        assert family.useful_change
        assert family.insufficient_change
        assert family.semantic_comparison_fields
        assert family.reenabling_recheck_triggers


def test_declarative_families_have_unique_semantic_signatures() -> None:
    signatures = {
        (
            family.invariant_principle.casefold().strip(),
            tuple(sorted(family.semantic_comparison_fields)),
        )
        for family in REPOSITORY.families()
    }

    assert len(signatures) == len(REPOSITORY.families())


@pytest.mark.parametrize(
    "family_id",
    ("macro_repeated_advice", "suppression_exact_duplicate"),
)
def test_meaningful_state_change_reenables_repeated_advice(
    family_id: str,
) -> None:
    scenario = build_scenario(REPOSITORY, REPOSITORY.require(family_id))
    changed = replace(
        scenario,
        context=replace(
            scenario.context, state_signature="meaningfully_changed"
        ),
        source_type="COUNTERFACTUAL",
    )

    assert ORACLE.decide(scenario).decision.decision == "SUPPRESS"
    assert ORACLE.decide(changed).decision.decision == "SHOW"


@pytest.mark.parametrize(
    "family_id",
    (
        "suppression_wording_only_change",
        "suppression_locale_only_change",
        "suppression_irrelevant_evidence_added",
    ),
)
def test_non_semantic_change_remains_suppressed(family_id: str) -> None:
    scenario = build_scenario(REPOSITORY, REPOSITORY.require(family_id))

    assert ORACLE.decide(scenario).decision.decision == "SUPPRESS"
