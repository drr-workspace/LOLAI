from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from collections.abc import Callable
from typing import Iterable, Mapping, Sequence

from generators.causal_signature import DatasetRecord
from generators.policy_loader import SplitPolicy, load_policies


SPLIT_NAMES = ("train", "valid", "test", "challenge")


class SplitConstraintError(ValueError):
    """Raised when cohesive records cannot satisfy the split constraints."""


@dataclass(frozen=True, slots=True)
class DistributionSummary:
    total: int
    tasks: Mapping[str, int]
    decisions: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class SplitResult:
    train: tuple[DatasetRecord, ...]
    valid: tuple[DatasetRecord, ...]
    test: tuple[DatasetRecord, ...]
    challenge: tuple[DatasetRecord, ...]
    summaries: Mapping[str, DistributionSummary]

    def named(self, split: str) -> tuple[DatasetRecord, ...]:
        if split not in SPLIT_NAMES:
            raise KeyError(split)
        return getattr(self, split)


class DatasetSplitter:
    def __init__(
        self,
        policy: SplitPolicy | None = None,
        *,
        seed: int = 0,
    ) -> None:
        self._policy = policy or load_policies().split
        self._seed = seed
        if self._policy.allow_random_per_row:
            raise SplitConstraintError(
                "La policy non può abilitare split casuali per riga"
            )

    def split(
        self,
        records: Sequence[DatasetRecord],
        *,
        reserved_test_families: frozenset[str] = frozenset(),
    ) -> SplitResult:
        if not records:
            raise SplitConstraintError("Nessun record da suddividere")
        units = self._cohesive_units(records)
        forced: list[tuple[DatasetRecord, ...]] = []
        free: list[tuple[DatasetRecord, ...]] = []
        for unit in units:
            families = {item.scenario.family_id for item in unit}
            (forced if families & reserved_test_families else free).append(
                unit
            )

        assigned: dict[str, list[DatasetRecord]] = {
            name: [] for name in SPLIT_NAMES
        }
        for unit in forced:
            assigned["test"].extend(unit)

        targets = {
            name: self._policy.distribution[name] * len(records)
            for name in SPLIT_NAMES
        }
        global_tasks = Counter(
            item.scenario.task for item in records
        )
        global_decisions = Counter(
            item.oracle_result.decision.decision for item in records
        )
        for unit in sorted(free, key=self._unit_order):
            eligible = tuple(
                name
                for name in SPLIT_NAMES
                if self._family_allowed(unit, name, assigned)
            )
            if not eligible:
                raise SplitConstraintError(
                    "Nessuno split compatibile per l'unità "
                    f"{unit[0].scenario.scenario_id}"
                )
            chosen = min(
                eligible,
                key=lambda name: (
                    self._assignment_cost(
                        unit,
                        assigned[name],
                        name,
                        targets,
                        global_tasks,
                        global_decisions,
                    ),
                    self._stable_hash(name),
                ),
            )
            assigned[chosen].extend(unit)

        self._validate_assignments(records, assigned, units)
        summaries = {
            name: _summary(items) for name, items in assigned.items()
        }
        return SplitResult(
            train=tuple(assigned["train"]),
            valid=tuple(assigned["valid"]),
            test=tuple(assigned["test"]),
            challenge=tuple(assigned["challenge"]),
            summaries=summaries,
        )

    def _assignment_cost(
        self,
        unit: tuple[DatasetRecord, ...],
        current: Sequence[DatasetRecord],
        split: str,
        size_targets: Mapping[str, float],
        global_tasks: Counter[str],
        global_decisions: Counter[str],
    ) -> float:
        probability = self._policy.distribution[split]
        projected_tasks = Counter(
            item.scenario.task for item in current
        )
        projected_tasks.update(item.scenario.task for item in unit)
        projected_decisions = Counter(
            item.oracle_result.decision.decision for item in current
        )
        projected_decisions.update(
            item.oracle_result.decision.decision for item in unit
        )
        current_tasks = Counter(item.scenario.task for item in current)
        current_decisions = Counter(
            item.oracle_result.decision.decision for item in current
        )
        size_cost = (
            len(current) + len(unit) - size_targets[split]
        ) ** 2 - (len(current) - size_targets[split]) ** 2
        task_cost = sum(
            (
                projected_tasks[task] - count * probability
            ) ** 2
            - (current_tasks[task] - count * probability) ** 2
            for task, count in global_tasks.items()
        )
        decision_cost = sum(
            (
                projected_decisions[decision] - count * probability
            ) ** 2
            - (
                current_decisions[decision] - count * probability
            ) ** 2
            for decision, count in global_decisions.items()
        )
        return size_cost + task_cost + decision_cost

    def _cohesive_units(
        self, records: Sequence[DatasetRecord]
    ) -> tuple[tuple[DatasetRecord, ...], ...]:
        parent = list(range(len(records)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        indexes: dict[tuple[str, str], int] = {}
        for index, record in enumerate(records):
            scenario = record.scenario
            keys = [("splitGroup", scenario.split_group)]
            if self._policy.prevent_train_test_overlap:
                keys.append(("familyId", scenario.family_id))
            if self._policy.keep_episodes_together and scenario.episode_id:
                keys.append(("episodeId", scenario.episode_id))
            if (
                self._policy.keep_counterfactual_pairs_together
                and scenario.counterfactual_pair_id
            ):
                keys.append(
                    (
                        "counterfactualPairId",
                        scenario.counterfactual_pair_id,
                    )
                )
            for key in keys:
                if key in indexes:
                    union(index, indexes[key])
                else:
                    indexes[key] = index

        groups: dict[int, list[DatasetRecord]] = {}
        for index, record in enumerate(records):
            groups.setdefault(find(index), []).append(record)
        return tuple(
            tuple(
                sorted(
                    group, key=lambda item: item.scenario.scenario_id
                )
            )
            for group in groups.values()
        )

    def _family_allowed(
        self,
        unit: tuple[DatasetRecord, ...],
        target: str,
        assigned: Mapping[str, list[DatasetRecord]],
    ) -> bool:
        if not self._policy.prevent_train_test_overlap:
            return True
        families = {item.scenario.family_id for item in unit}
        opposite = (
            assigned["test"]
            if target == "train"
            else assigned["train"]
            if target == "test"
            else ()
        )
        return families.isdisjoint(
            item.scenario.family_id for item in opposite
        )

    def _unit_order(
        self, unit: tuple[DatasetRecord, ...]
    ) -> tuple[int, str]:
        identity = "|".join(
            item.scenario.scenario_id for item in unit
        )
        return (-len(unit), self._stable_hash(identity))

    def _stable_hash(self, value: str) -> str:
        return hashlib.sha256(
            f"{self._seed}:{value}".encode("utf-8")
        ).hexdigest()

    def _validate_assignments(
        self,
        original: Sequence[DatasetRecord],
        assigned: Mapping[str, list[DatasetRecord]],
        units: Sequence[tuple[DatasetRecord, ...]],
    ) -> None:
        flattened = [item for split in SPLIT_NAMES for item in assigned[split]]
        if len(flattened) != len(original):
            raise SplitConstraintError("Record persi o duplicati nello split")
        self._validate_cohesion(assigned, "episode_id")
        self._validate_cohesion(assigned, "counterfactual_pair_id")
        if self._policy.prevent_train_test_overlap:
            train_families = {
                item.scenario.family_id for item in assigned["train"]
            }
            test_families = {
                item.scenario.family_id for item in assigned["test"]
            }
            overlap = train_families & test_families
            if overlap:
                raise SplitConstraintError(
                    f"Famiglie condivise tra train e test: {sorted(overlap)}"
                )
        self._validate_distributions(original, assigned, units)

    def _validate_distributions(
        self,
        original: Sequence[DatasetRecord],
        assigned: Mapping[str, list[DatasetRecord]],
        units: Sequence[tuple[DatasetRecord, ...]],
    ) -> None:
        largest_unit = max(len(unit) for unit in units)
        global_tasks = Counter(item.scenario.task for item in original)
        global_decisions = Counter(
            item.oracle_result.decision.decision for item in original
        )
        for split, records in assigned.items():
            probability = self._policy.distribution[split]
            expected_size = len(original) * probability
            if abs(len(records) - expected_size) > largest_unit:
                raise SplitConstraintError(
                    f"Distribuzione totale irrispettabile per {split}: "
                    f"attesi {expected_size:.2f}, ottenuti {len(records)}"
                )
            actual_tasks = Counter(
                item.scenario.task for item in records
            )
            actual_decisions = Counter(
                item.oracle_result.decision.decision for item in records
            )
            self._validate_category_distribution(
                split,
                "task",
                global_tasks,
                actual_tasks,
                probability,
                units,
                lambda item: item.scenario.task,
            )
            self._validate_category_distribution(
                split,
                "decision",
                global_decisions,
                actual_decisions,
                probability,
                units,
                lambda item: item.oracle_result.decision.decision,
            )

    @staticmethod
    def _validate_category_distribution(
        split: str,
        category: str,
        global_counts: Counter[str],
        actual_counts: Counter[str],
        probability: float,
        units: Sequence[tuple[DatasetRecord, ...]],
        value_of: Callable[[DatasetRecord], str],
    ) -> None:
        for value, count in global_counts.items():
            largest_atomic_count = max(
                sum(
                    1
                    for item in unit
                    if value_of(item) == value
                )
                for unit in units
            )
            expected = count * probability
            if (
                abs(actual_counts[value] - expected)
                > largest_atomic_count
            ):
                raise SplitConstraintError(
                    f"Distribuzione {category} irrispettabile in {split} "
                    f"per {value}: attesi {expected:.2f}, "
                    f"ottenuti {actual_counts[value]}"
                )

    @staticmethod
    def _validate_cohesion(
        assigned: Mapping[str, list[DatasetRecord]], field: str
    ) -> None:
        locations: dict[str, str] = {}
        for split, records in assigned.items():
            for record in records:
                value = getattr(record.scenario, field)
                if value is None:
                    continue
                previous = locations.setdefault(value, split)
                if previous != split:
                    raise SplitConstraintError(
                        f"{field} {value!r} diviso tra split"
                    )


def split_dataset(
    records: Sequence[DatasetRecord],
    *,
    seed: int = 0,
    reserved_test_families: frozenset[str] = frozenset(),
    policy: SplitPolicy | None = None,
) -> SplitResult:
    return DatasetSplitter(policy, seed=seed).split(
        records, reserved_test_families=reserved_test_families
    )


def _summary(records: Iterable[DatasetRecord]) -> DistributionSummary:
    items = tuple(records)
    return DistributionSummary(
        total=len(items),
        tasks=dict(Counter(item.scenario.task for item in items)),
        decisions=dict(
            Counter(
                item.oracle_result.decision.decision for item in items
            )
        ),
    )
