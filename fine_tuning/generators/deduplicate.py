from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Sequence

from generators.causal_signature import (
    CausalSignatureBuilder,
    DatasetRecord,
)
from generators.policy_loader import (
    DeduplicationPolicy,
    load_policies,
)


@dataclass(frozen=True, slots=True)
class Removal:
    scenario_ids: tuple[str, ...]
    reason: str
    retained_scenario_id: str | None
    causal_signature: str
    similarity: float | None = None


@dataclass(frozen=True, slots=True)
class DeduplicationReport:
    input_count: int
    kept: tuple[DatasetRecord, ...]
    removals: tuple[Removal, ...]

    @property
    def removed_count(self) -> int:
        return self.input_count - len(self.kept)


class Deduplicator:
    def __init__(self, policy: DeduplicationPolicy | None = None) -> None:
        self._policy = policy or load_policies().deduplication
        self._signatures = CausalSignatureBuilder(self._policy)

    def run(
        self,
        records: Sequence[DatasetRecord],
        *,
        progress: Callable[[int, int, int], None] | None = None,
    ) -> DeduplicationReport:
        units = self._atomic_units(records)
        kept: list[DatasetRecord] = []
        kept_by_structure: dict[
            tuple[str, str], list[DatasetRecord]
        ] = {}
        removals: list[Removal] = []
        exact_seen: dict[str, DatasetRecord] = {}
        signature_counts: dict[str, int] = {}

        for unit_index, unit in enumerate(units, start=1):
            if progress is not None and unit_index % 5_000 == 0:
                progress(unit_index, len(units), len(kept))
            protected = any(_is_human_approved(item) for item in unit)
            representative = unit[0]
            signature = self._signatures.build(representative).digest
            exact = _exact_fingerprint(unit)
            retained: DatasetRecord | None = exact_seen.get(exact)
            reason: str | None = None
            similarity: float | None = None

            if retained is not None and not protected:
                reason = "EXACT_DUPLICATE"
            elif (
                signature_counts.get(signature, 0) + len(unit)
                > self._policy.maximum_examples_per_signature
                and not protected
            ):
                reason = "CAUSAL_SIGNATURE_LIMIT"
            elif not protected:
                structural_key = (
                    representative.scenario.task,
                    self._signatures.structural_key(representative),
                )
                retained, similarity = self._find_similar(
                    representative,
                    kept_by_structure.get(structural_key, ()),
                )
                if retained is not None:
                    reason = "STRUCTURAL_SIMILARITY"

            if reason is not None:
                removals.append(
                    Removal(
                        scenario_ids=tuple(
                            item.scenario.scenario_id for item in unit
                        ),
                        reason=reason,
                        retained_scenario_id=(
                            retained.scenario.scenario_id
                            if retained is not None
                            else None
                        ),
                        causal_signature=signature,
                        similarity=similarity,
                    )
                )
                continue

            kept.extend(unit)
            for item in unit:
                structural_key = (
                    item.scenario.task,
                    self._signatures.structural_key(item),
                )
                kept_by_structure.setdefault(
                    structural_key, []
                ).append(item)
            exact_seen[exact] = representative
            signature_counts[signature] = (
                signature_counts.get(signature, 0) + len(unit)
            )
        if progress is not None:
            progress(len(units), len(units), len(kept))

        return DeduplicationReport(
            input_count=len(records),
            kept=tuple(kept),
            removals=tuple(removals),
        )

    def _atomic_units(
        self, records: Sequence[DatasetRecord]
    ) -> tuple[tuple[DatasetRecord, ...], ...]:
        grouped_members: dict[
            tuple[str, str], list[DatasetRecord]
        ] = {}
        singles: list[tuple[DatasetRecord, ...]] = []
        for record in records:
            scenario = record.scenario
            if scenario.episode_id:
                grouped_members.setdefault(
                    ("episode", scenario.episode_id), []
                ).append(record)
            elif scenario.counterfactual_pair_id:
                grouped_members.setdefault(
                    ("pair", scenario.counterfactual_pair_id), []
                ).append(record)
            else:
                singles.append((record,))
        groups = [
            tuple(
                sorted(
                    members,
                    key=lambda item: item.scenario.scenario_id,
                )
            )
            for _, members in sorted(grouped_members.items())
        ]
        return tuple(
            sorted(
                [*singles, *groups],
                key=lambda unit: min(
                    item.scenario.scenario_id for item in unit
                ),
            )
        )

    def _find_similar(
        self,
        candidate: DatasetRecord,
        kept: Iterable[DatasetRecord],
    ) -> tuple[DatasetRecord | None, float | None]:
        candidate_signature = self._signatures.build(candidate).digest
        candidate_features = self._signatures.structural_features(candidate)
        best: DatasetRecord | None = None
        best_score = 0.0
        for existing in kept:
            if (
                self._signatures.build(existing).digest
                == candidate_signature
            ):
                continue
            existing_features = self._signatures.structural_features(existing)
            score = structural_similarity(
                candidate_features, existing_features
            )
            if score > best_score:
                best, best_score = existing, score
        if best_score >= self._policy.structural_similarity_threshold:
            return best, best_score
        return None, None


def structural_similarity(
    left: frozenset[str], right: frozenset[str]
) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def deduplicate(
    records: Sequence[DatasetRecord],
    policy: DeduplicationPolicy | None = None,
    *,
    progress: Callable[[int, int, int], None] | None = None,
) -> DeduplicationReport:
    return Deduplicator(policy).run(records, progress=progress)


def _is_human_approved(record: DatasetRecord) -> bool:
    return record.review_status == "APPROVED"


def _exact_fingerprint(unit: tuple[DatasetRecord, ...]) -> str:
    payload = [
        {
            "scenario": asdict(record.scenario),
            "decision": asdict(record.oracle_result.decision),
            "counterfactualVariable": record.counterfactual_variable,
        }
        for record in unit
    ]
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _json_default(value: object) -> object:
    if isinstance(value, (set, frozenset, tuple)):
        return sorted(value)
    raise TypeError(f"Tipo non serializzabile: {type(value).__name__}")
