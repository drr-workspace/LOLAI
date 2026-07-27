from __future__ import annotations

import random
from dataclasses import replace

import pytest

from generators.causal_signature import (
    CausalSignatureBuilder,
    DatasetRecord,
)
from generators.id_factory import IdFactory
from generators.oracle import StrategicOracle
from generators.random_source import RandomSource
from generators.sampler import ScenarioSampler


SAMPLER = ScenarioSampler()
ORACLE = StrategicOracle()
FAMILY_ID = "composition_protect_vs_dive"


def test_random_source_is_deterministic() -> None:
    first = RandomSource(17)
    second = RandomSource(17)

    assert [first.randint(1, 100) for _ in range(8)] == [
        second.randint(1, 100) for _ in range(8)
    ]


def test_random_source_does_not_touch_global_random_state() -> None:
    random.seed(912)
    expected = random.random()
    random.seed(912)

    source = RandomSource(4)
    source.randint(0, 100)

    assert random.random() == expected


def test_same_seed_produces_same_scenario() -> None:
    first = SAMPLER.sample(FAMILY_ID, 81, output_locale="it-IT")
    second = SAMPLER.sample(FAMILY_ID, 81, output_locale="it-IT")

    assert first == second


def test_scenario_records_required_metadata() -> None:
    scenario = SAMPLER.sample(FAMILY_ID, 82, ordinal=3)

    assert scenario.family_id == FAMILY_ID
    assert scenario.split_group
    assert scenario.seed == 82
    assert len(scenario.causal_signature) == 64


def test_candidate_position_varies_across_seeds() -> None:
    selected_positions: set[int] = set()
    for seed in range(40, 80):
        scenario = SAMPLER.sample(FAMILY_ID, seed)
        result = ORACLE.decide(scenario)
        if result.decision.primary_action_id is not None:
            selected_positions.add(
                tuple(
                    candidate.action_id for candidate in scenario.candidates
                ).index(result.decision.primary_action_id)
            )

    assert len(selected_positions) > 1


def test_sample_sizes_vary_within_policy_bounds() -> None:
    sizes = {
        (
            len(SAMPLER.sample(FAMILY_ID, seed).candidates),
            len(SAMPLER.sample(FAMILY_ID, seed).evidence),
            len(SAMPLER.sample(FAMILY_ID, seed).threats),
        )
        for seed in range(20, 45)
    }

    assert len(sizes) > 1
    assert all(1 <= candidates <= 5 for candidates, _, _ in sizes)
    assert all(1 <= evidence <= 10 for _, evidence, _ in sizes)


def test_declared_causal_parameters_create_signature_capacity() -> None:
    signatures: set[str] = set()
    builder = CausalSignatureBuilder()
    for ordinal in range(1, 126):
        scenario = SAMPLER.sample(
            FAMILY_ID,
            5_000 + ordinal,
            ordinal=ordinal,
        )
        result = ORACLE.decide(scenario)
        signatures.add(
            builder.build(DatasetRecord(scenario, result)).digest
        )

    assert len(signatures) >= 50


def test_id_rename_preserves_semantic_oracle_result() -> None:
    scenario = SAMPLER.sample(FAMILY_ID, 101)
    renamed = IdFactory(999, style="misleading").rename_scenario(scenario)
    original = ORACLE.decide(scenario)
    changed = ORACLE.decide(renamed)

    assert original.decision.decision == changed.decision.decision
    assert original.decision.confidence == changed.decision.confidence
    if original.decision.primary_action_id is not None:
        original_candidate = next(
            item
            for item in scenario.candidates
            if item.action_id == original.decision.primary_action_id
        )
        renamed_candidate = next(
            item
            for item in renamed.candidates
            if item.action_id == changed.decision.primary_action_id
        )
        assert (
            original_candidate.equivalence_key
            == renamed_candidate.equivalence_key
        )


def test_candidate_order_preserves_decision() -> None:
    scenario = SAMPLER.sample(FAMILY_ID, 102)
    reordered = replace(
        scenario, candidates=tuple(reversed(scenario.candidates))
    )

    assert ORACLE.decide(scenario).decision == ORACLE.decide(
        reordered
    ).decision


@pytest.mark.parametrize("locale", ("it-IT", "en-US"))
def test_supported_locales(locale: str) -> None:
    assert SAMPLER.sample(FAMILY_ID, 103, output_locale=locale).output_locale == locale


def test_unsupported_locale_is_rejected() -> None:
    with pytest.raises(ValueError, match="locale"):
        SAMPLER.sample(FAMILY_ID, 104, output_locale="fr-FR")


@pytest.mark.parametrize(
    "style", ("neutral", "compact", "misleading")
)
def test_id_factory_styles_are_deterministic(style: str) -> None:
    first = IdFactory(7, style=style)
    second = IdFactory(7, style=style)

    assert first.action_id("candidate") == second.action_id("candidate")
    assert first.evidence_id("signal") != first.entity_id("signal")
