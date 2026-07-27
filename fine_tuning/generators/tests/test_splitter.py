from __future__ import annotations

from dataclasses import replace

from generators.causal_signature import DatasetRecord
from generators.oracle import StrategicOracle
from generators.sampler import ScenarioSampler
from generators.splitter import SPLIT_NAMES, split_dataset


SAMPLER = ScenarioSampler()
ORACLE = StrategicOracle()
FAMILIES = (
    "composition_protect_vs_dive",
    "matchup_break_enemy_freeze",
    "itemization_dominant_physical_threat",
    "threat_effective_vs_theoretical_damage",
    "macro_objective_setup",
    "suppression_exact_duplicate",
)


def _records(count: int = 24) -> tuple[DatasetRecord, ...]:
    records: list[DatasetRecord] = []
    for index in range(count):
        family = FAMILIES[index % len(FAMILIES)]
        scenario = SAMPLER.sample(
            family,
            800 + index,
            ordinal=index,
        )
        scenario = replace(
            scenario,
            split_group=f"group_{index}",
        )
        records.append(DatasetRecord(scenario, ORACLE.decide(scenario)))
    return tuple(records)


def _locations(result: object) -> dict[str, str]:
    return {
        record.scenario.scenario_id: split
        for split in SPLIT_NAMES
        for record in getattr(result, split)
    }


def test_same_seed_produces_same_splits() -> None:
    records = _records()

    first = split_dataset(records, seed=91)
    second = split_dataset(tuple(reversed(records)), seed=91)

    assert {
        name: {
            item.scenario.scenario_id for item in first.named(name)
        }
        for name in SPLIT_NAMES
    } == {
        name: {
            item.scenario.scenario_id for item in second.named(name)
        }
        for name in SPLIT_NAMES
    }


def test_episode_is_not_split() -> None:
    records = list(_records())
    records[0] = replace(
        records[0],
        scenario=replace(
            records[0].scenario,
            episode_id="episode_shared",
            split_group="episode_group_a",
        ),
    )
    records[1] = replace(
        records[1],
        scenario=replace(
            records[1].scenario,
            episode_id="episode_shared",
            split_group="episode_group_b",
        ),
    )

    locations = _locations(split_dataset(tuple(records), seed=92))

    assert locations[records[0].scenario.scenario_id] == locations[
        records[1].scenario.scenario_id
    ]


def test_counterfactual_pair_is_not_split() -> None:
    records = list(_records())
    for index in (2, 3):
        records[index] = replace(
            records[index],
            scenario=replace(
                records[index].scenario,
                counterfactual_pair_id="pair_shared",
                split_group=f"pair_group_{index}",
            ),
        )

    locations = _locations(split_dataset(tuple(records), seed=93))

    assert locations[records[2].scenario.scenario_id] == locations[
        records[3].scenario.scenario_id
    ]


def test_split_group_is_not_split() -> None:
    records = list(_records())
    for index in (4, 5):
        records[index] = replace(
            records[index],
            scenario=replace(
                records[index].scenario, split_group="shared_group"
            ),
        )

    locations = _locations(split_dataset(tuple(records), seed=94))

    assert locations[records[4].scenario.scenario_id] == locations[
        records[5].scenario.scenario_id
    ]


def test_reserved_test_family_never_appears_in_train() -> None:
    records = _records()
    reserved = frozenset({FAMILIES[0]})
    result = split_dataset(
        records, seed=95, reserved_test_families=reserved
    )

    assert all(
        item.scenario.family_id not in reserved for item in result.train
    )
    assert any(
        item.scenario.family_id in reserved for item in result.test
    )


def test_train_and_test_families_do_not_overlap() -> None:
    result = split_dataset(_records(36), seed=96)

    train = {item.scenario.family_id for item in result.train}
    test = {item.scenario.family_id for item in result.test}

    assert train.isdisjoint(test)


def test_summaries_cover_tasks_and_decisions() -> None:
    result = split_dataset(_records(36), seed=97)

    for split in SPLIT_NAMES:
        summary = result.summaries[split]
        assert summary.total == len(result.named(split))
        assert sum(summary.tasks.values()) == summary.total
        assert sum(summary.decisions.values()) == summary.total
