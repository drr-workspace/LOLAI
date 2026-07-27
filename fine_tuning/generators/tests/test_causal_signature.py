from __future__ import annotations

from dataclasses import replace

from generators.causal_signature import (
    DatasetRecord,
    build_causal_signature,
)
from generators.id_factory import IdFactory
from generators.oracle import StrategicOracle
from generators.sampler import ScenarioSampler


SAMPLER = ScenarioSampler()
ORACLE = StrategicOracle()
FAMILY = "itemization_dominant_physical_threat"


def _sample(seed: int = 610) -> DatasetRecord:
    scenario = SAMPLER.sample(FAMILY, seed)
    return DatasetRecord(scenario, ORACLE.decide(scenario))


def test_renaming_ids_produces_same_signature() -> None:
    original = _sample()
    renamed_scenario = IdFactory(99, style="misleading").rename_scenario(
        original.scenario
    )
    renamed = DatasetRecord(
        renamed_scenario, ORACLE.decide(renamed_scenario)
    )

    assert build_causal_signature(
        original.scenario, original.oracle_result
    ).digest == build_causal_signature(
        renamed.scenario, renamed.oracle_result
    ).digest


def test_candidate_and_evidence_order_do_not_change_signature() -> None:
    original = _sample(611)
    reordered_scenario = replace(
        original.scenario,
        candidates=tuple(reversed(original.scenario.candidates)),
        evidence=tuple(reversed(original.scenario.evidence)),
    )
    reordered = ORACLE.decide(reordered_scenario)

    assert build_causal_signature(
        original.scenario, original.oracle_result
    ).digest == build_causal_signature(
        reordered_scenario, reordered
    ).digest


def test_small_numeric_change_stays_in_same_bucket() -> None:
    original = _sample(612)
    context = replace(
        original.scenario.context,
        completeness=min(
            0.999, original.scenario.context.completeness + 0.001
        ),
    )
    changed = replace(original.scenario, context=context)

    assert build_causal_signature(
        original.scenario, original.oracle_result
    ).digest == build_causal_signature(
        changed, ORACLE.decide(changed)
    ).digest


def test_counterfactual_variable_changes_signature() -> None:
    record = _sample(613)

    first = build_causal_signature(
        record.scenario,
        record.oracle_result,
        counterfactual_variable="candidateFeasibility",
    )
    second = build_causal_signature(
        record.scenario,
        record.oracle_result,
        counterfactual_variable="evidenceFreshness",
    )

    assert first.digest != second.digest


def test_reason_codes_change_signature() -> None:
    record = _sample(616)
    changed_result = replace(
        record.oracle_result,
        decision=replace(
            record.oracle_result.decision,
            reason_codes=("LOW_ACTION_VALUE",),
        ),
        trace=replace(
            record.oracle_result.trace,
            reason_codes=("LOW_ACTION_VALUE",),
        ),
    )

    assert build_causal_signature(
        record.scenario, record.oracle_result
    ).digest != build_causal_signature(
        record.scenario, changed_result
    ).digest


def test_causal_feasibility_change_changes_signature() -> None:
    original = _sample(614)
    candidate = original.scenario.candidates[0]
    changed_scenario = replace(
        original.scenario,
        candidates=(
            replace(candidate, feasibility=0.0),
            *original.scenario.candidates[1:],
        ),
    )

    assert build_causal_signature(
        original.scenario, original.oracle_result
    ).digest != build_causal_signature(
        changed_scenario,
        ORACLE.decide(changed_scenario),
        counterfactual_variable="candidateFeasibility",
    ).digest


def test_message_formulation_is_not_part_of_signature() -> None:
    record = _sample(615)
    components = build_causal_signature(
        record.scenario, record.oracle_result
    ).as_dict()

    assert "message" not in components
