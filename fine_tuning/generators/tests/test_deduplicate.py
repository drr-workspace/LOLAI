from __future__ import annotations

from dataclasses import replace

from generators.causal_signature import DatasetRecord
from generators.deduplicate import deduplicate
from generators.episodes import EpisodeGenerator
from generators.id_factory import IdFactory
from generators.oracle import StrategicOracle
from generators.sampler import ScenarioSampler


SAMPLER = ScenarioSampler()
ORACLE = StrategicOracle()
FAMILY = "threat_effective_vs_theoretical_damage"


def _record(
    seed: int,
    *,
    scenario_id: str | None = None,
    pair_id: str | None = None,
    review_status: str = "GENERATED",
) -> DatasetRecord:
    scenario = SAMPLER.sample(FAMILY, seed)
    scenario = replace(
        scenario,
        scenario_id=scenario_id or scenario.scenario_id,
        counterfactual_pair_id=pair_id,
    )
    return DatasetRecord(
        scenario, ORACLE.decide(scenario), review_status=review_status
    )


def test_exact_duplicate_is_removed() -> None:
    record = _record(701)
    report = deduplicate((record, record))

    assert len(report.kept) == 1
    assert report.removals[0].reason == "EXACT_DUPLICATE"


def test_structurally_similar_record_is_removed() -> None:
    first = _record(702, scenario_id="scenario_a")
    second_scenario = replace(
        first.scenario,
        scenario_id="scenario_b",
        seed=999,
    )
    second = DatasetRecord(
        second_scenario,
        ORACLE.decide(second_scenario),
        counterfactual_variable="metadata_only_axis",
    )

    report = deduplicate((first, second))

    assert len(report.kept) == 1
    assert report.removals[0].reason == "STRUCTURAL_SIMILARITY"


def test_distinct_causal_buckets_are_preserved() -> None:
    family = "composition_protect_vs_dive"
    first_scenario = SAMPLER.sample(family, 801, ordinal=1)
    second_scenario = SAMPLER.sample(family, 802, ordinal=24)
    first = DatasetRecord(first_scenario, ORACLE.decide(first_scenario))
    second = DatasetRecord(second_scenario, ORACLE.decide(second_scenario))

    report = deduplicate((first, second))

    assert len(report.kept) == 2


def test_signature_limit_allows_four_non_exact_variants() -> None:
    original = _record(804)
    records = []
    for index in range(5):
        renamed = IdFactory(900 + index).rename_scenario(
            original.scenario
        )
        renamed = replace(
            renamed,
            scenario_id=f"signature_variant_{index}",
            seed=900 + index,
        )
        records.append(
            DatasetRecord(renamed, ORACLE.decide(renamed))
        )

    report = deduplicate(tuple(records))

    assert len(report.kept) == 4
    assert report.removals[0].reason == "CAUSAL_SIGNATURE_LIMIT"


def test_human_approved_record_is_preserved() -> None:
    generated = _record(703)
    approved = replace(generated, review_status="APPROVED")

    report = deduplicate((generated, approved))

    assert approved in report.kept
    assert len(report.kept) == 2


def test_counterfactual_pair_is_kept_or_removed_as_a_unit() -> None:
    first = _record(704, scenario_id="pair_a", pair_id="pair_1")
    second = _record(705, scenario_id="pair_b", pair_id="pair_1")

    report = deduplicate((first, second))
    kept_ids = {item.scenario.scenario_id for item in report.kept}

    assert kept_ids in (set(), {"pair_a", "pair_b"})


def test_temporal_episode_is_kept_or_removed_as_a_unit() -> None:
    episode = EpisodeGenerator().generate(
        "episode_vision_contest_basic", 1_404
    )
    records = tuple(
        DatasetRecord(step.scenario, step.oracle_result)
        for step in episode.steps
    )

    report = deduplicate(records)

    assert len(report.kept) in (0, len(records))


def test_report_counts_removed_records() -> None:
    record = _record(706)
    report = deduplicate((record, record, record))

    assert report.input_count == 3
    assert report.removed_count == 2
    assert sum(len(item.scenario_ids) for item in report.removals) == 2
