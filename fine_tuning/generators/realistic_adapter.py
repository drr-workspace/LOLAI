from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from generators.domain_models import (
    CandidateAction,
    CanonicalScenario,
    Evidence,
    OracleResult,
    RecentAdvice,
    ScenarioContext,
    TeamPlan,
    Threat,
)
from generators.ontology_registry import (
    OntologyRegistry,
    load_ontology,
)
from generators.oracle import StrategicOracle


SCHEMA_VERSION = "1.0.0"
REVIEW_STATUS = "NEEDS_REVIEW"


class SnapshotValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AdaptedScenario:
    scenario: CanonicalScenario
    oracle_result: OracleResult
    review_status: str
    provenance: Mapping[str, object]


def _schema_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "contracts" / "task-schemas"


def _load_schema(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise SnapshotValidationError(f"{path}: schema non valido")
    return value


def _validator() -> Draft202012Validator:
    schema_dir = _schema_dir()
    common = _load_schema(schema_dir / "common.schema.json")
    snapshot = _load_schema(
        schema_dir / "abstract-runtime-snapshot.schema.json"
    )
    registry = Registry().with_resources(
        [
            ("common.schema.json", Resource.from_contents(common)),
            (
                "abstract-runtime-snapshot.schema.json",
                Resource.from_contents(snapshot),
            ),
        ]
    )
    Draft202012Validator.check_schema(snapshot)
    return Draft202012Validator(
        snapshot,
        registry=registry,
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    )


def _unit(value: object, location: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SnapshotValidationError(f"{location}: valore numerico atteso")
    numeric = float(value)
    normalized = numeric / 100.0 if numeric > 1.0 else numeric
    if not 0.0 <= normalized <= 1.0:
        raise SnapshotValidationError(f"{location}: valore fuori scala")
    return round(normalized, 6)


def _strings(value: object, location: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise SnapshotValidationError(f"{location}: lista attesa")
    result = tuple(str(item) for item in value)
    if len(result) != len(set(result)):
        raise SnapshotValidationError(f"{location}: valori duplicati")
    return result


def _require_ids(
    registry: OntologyRegistry,
    registry_id: str,
    values: Sequence[str],
    location: str,
) -> None:
    unknown = set(values) - registry.ids(registry_id)
    if unknown:
        raise SnapshotValidationError(
            f"{location}: valori non ontologici {sorted(unknown)}"
        )


def _opaque_seed(snapshot_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(snapshot_id.encode("utf-8")).digest()[:8], "big"
    ) % (2**31)


def _signature(
    task: str,
    context: ScenarioContext,
    candidates: Sequence[CandidateAction],
    threats: Sequence[Threat],
    team_plan: TeamPlan,
) -> str:
    value = {
        "task": task,
        "completeness": context.completeness,
        "freshness": context.freshness_seconds,
        "candidateTypes": sorted(item.action_type for item in candidates),
        "candidateFeasibility": sorted(item.feasibility for item in candidates),
        "threatPatterns": sorted(
            pattern for item in threats for pattern in item.patterns
        ),
        "threatPriority": sorted(item.priority for item in threats),
        "missingFunctions": sorted(team_plan.missing_functions),
    }
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class RealisticAdapter:
    def __init__(
        self,
        ontology: OntologyRegistry | None = None,
        oracle: StrategicOracle | None = None,
    ) -> None:
        self._ontology = ontology or load_ontology()
        self._oracle = oracle or StrategicOracle(ontology=self._ontology)
        self._validator = _validator()

    def validate(self, snapshot: Mapping[str, object]) -> None:
        errors = sorted(
            self._validator.iter_errors(snapshot),
            key=lambda error: tuple(
                str(part) for part in error.absolute_path
            ),
        )
        if errors:
            first = errors[0]
            path = "$" + "".join(
                f"[{part!r}]" for part in first.absolute_path
            )
            raise SnapshotValidationError(f"{path}: {first.message}")
        if snapshot["schemaVersion"] != SCHEMA_VERSION:
            raise SnapshotValidationError("schemaVersion non supportata")
        task = str(snapshot["task"])
        if not self._ontology.contains("tasks", task):
            raise SnapshotValidationError(f"task non ontologico: {task}")
        team = cast(Mapping[str, object], snapshot["team"])
        for key in (
            "allyFunctions",
            "enemyFunctions",
            "missingFunctions",
        ):
            values = _strings(team[key], f"team.{key}")
            _require_ids(
                self._ontology, "team-functions", values, f"team.{key}"
            )
        for key in ("allyArchetypes", "enemyArchetypes"):
            values = _strings(team[key], f"team.{key}")
            _require_ids(
                self._ontology, "archetypes", values, f"team.{key}"
            )
        primary = str(team["primaryWinCondition"])
        if not self._ontology.contains("archetypes", primary):
            raise SnapshotValidationError(
                f"primaryWinCondition non ontologica: {primary}"
            )
        waves = cast(Mapping[str, object], snapshot["waveStates"])
        _require_ids(
            self._ontology,
            "wave-states",
            tuple(str(value) for value in waves.values()),
            "waveStates",
        )
        for index, raw in enumerate(
            cast(Sequence[Mapping[str, object]], snapshot["threats"])
        ):
            patterns = _strings(raw["patterns"], f"threats[{index}].patterns")
            _require_ids(
                self._ontology,
                "threat-patterns",
                patterns,
                f"threats[{index}].patterns",
            )
        for index, raw in enumerate(
            cast(Sequence[Mapping[str, object]], snapshot["candidates"])
        ):
            action_type = str(raw["type"])
            if not self._ontology.contains("action-types", action_type):
                raise SnapshotValidationError(
                    f"candidates[{index}].type non ontologico: {action_type}"
                )
            functions = _strings(
                raw["supportsFunctions"],
                f"candidates[{index}].supportsFunctions",
            )
            _require_ids(
                self._ontology,
                "team-functions",
                functions,
                f"candidates[{index}].supportsFunctions",
            )
        for index, raw in enumerate(
            cast(Sequence[Mapping[str, object]], snapshot["evidence"])
        ):
            evidence_type = str(raw["type"])
            if not self._ontology.contains(
                "evidence-types", evidence_type
            ):
                raise SnapshotValidationError(
                    f"evidence[{index}].type non ontologico: "
                    f"{evidence_type}"
                )
            if evidence_type == "PATCH_KNOWLEDGE":
                raise SnapshotValidationError(
                    f"evidence[{index}]: PATCH_KNOWLEDGE non ammesso"
                )
        self._validate_references(snapshot)

    def _validate_references(
        self, snapshot: Mapping[str, object]
    ) -> None:
        evidence = cast(
            Sequence[Mapping[str, object]], snapshot["evidence"]
        )
        candidates = cast(
            Sequence[Mapping[str, object]], snapshot["candidates"]
        )
        threats = cast(
            Sequence[Mapping[str, object]], snapshot["threats"]
        )
        evidence_ids = [str(item["evidenceId"]) for item in evidence]
        action_ids = [str(item["actionId"]) for item in candidates]
        entity_ids = [str(item["entityId"]) for item in threats]
        for label, values in (
            ("evidenceId", evidence_ids),
            ("actionId", action_ids),
            ("entityId", entity_ids),
        ):
            if len(values) != len(set(values)):
                raise SnapshotValidationError(f"{label} duplicato")
        known_evidence = set(evidence_ids)
        known_entities = set(entity_ids)
        for index, item in enumerate(candidates):
            unknown_evidence = set(
                cast(Sequence[str], item["evidenceIds"])
            ) - known_evidence
            unknown_threats = set(
                cast(Sequence[str], item["counteredThreatIds"])
            ) - known_entities
            if unknown_evidence or unknown_threats:
                raise SnapshotValidationError(
                    f"candidates[{index}]: riferimenti sconosciuti"
                )
        for index, item in enumerate(threats):
            if set(
                cast(Sequence[str], item["evidenceIds"])
            ) - known_evidence:
                raise SnapshotValidationError(
                    f"threats[{index}]: evidenceId sconosciuto"
                )
        for index, item in enumerate(evidence):
            if set(
                cast(Sequence[str], item["conflictsWithEvidenceIds"])
            ) - known_evidence:
                raise SnapshotValidationError(
                    f"evidence[{index}]: conflitto sconosciuto"
                )

    def adapt(self, snapshot: Mapping[str, object]) -> AdaptedScenario:
        self.validate(snapshot)
        context_raw = cast(Mapping[str, object], snapshot["context"])
        team_raw = cast(Mapping[str, object], snapshot["team"])
        evidence_raw = cast(
            Sequence[Mapping[str, object]], snapshot["evidence"]
        )
        threats_raw = cast(
            Sequence[Mapping[str, object]], snapshot["threats"]
        )
        candidates_raw = cast(
            Sequence[Mapping[str, object]], snapshot["candidates"]
        )
        recent_raw = cast(
            Sequence[Mapping[str, object]], snapshot["recentAdvice"]
        )
        evidence = tuple(
            Evidence(
                evidence_id=str(item["evidenceId"]),
                category=str(item["type"]),
                confidence=_unit(
                    item["confidence"], f"evidence[{index}].confidence"
                ),
                freshness_seconds=int(item["freshnessSeconds"]),
                conflicts_with_evidence_ids=frozenset(
                    cast(Sequence[str], item["conflictsWithEvidenceIds"])
                ),
                fact=item["fact"],
            )
            for index, item in enumerate(evidence_raw)
        )
        threats: list[Threat] = []
        for index, item in enumerate(threats_raw):
            damage = cast(Mapping[str, object], item["damageProfile"])
            components = tuple(
                float(damage[key]) for key in ("physical", "magic", "true")
            )
            total = sum(components)
            if total <= 0:
                raise SnapshotValidationError(
                    f"threats[{index}].damageProfile: somma nulla"
                )
            threats.append(
                Threat(
                    entity_id=str(item["entityId"]),
                    priority=_unit(
                        item["priority"], f"threats[{index}].priority"
                    ),
                    evidence_ids=tuple(
                        cast(Sequence[str], item["evidenceIds"])
                    ),
                    patterns=tuple(
                        cast(Sequence[str], item["patterns"])
                    ),
                    damage_profile=cast(
                        tuple[float, float, float],
                        tuple(round(value / total, 6) for value in components),
                    ),
                )
            )
        candidates = tuple(
            CandidateAction(
                action_id=str(item["actionId"]),
                action_type=str(item["type"]),
                evidence_ids=tuple(
                    cast(Sequence[str], item["evidenceIds"])
                ),
                feasibility=_unit(
                    item["feasibility"],
                    f"candidates[{index}].feasibility",
                ),
                supports_functions=frozenset(
                    cast(Sequence[str], item["supportsFunctions"])
                ),
                countered_threat_ids=frozenset(
                    cast(Sequence[str], item["counteredThreatIds"])
                ),
                win_condition_tags=frozenset(
                    cast(Sequence[str], item["winConditionTags"])
                ),
                urgency_alignment=_unit(
                    item["urgencyAlignment"],
                    f"candidates[{index}].urgencyAlignment",
                ),
                opportunity_cost=_unit(
                    item["opportunityCost"],
                    f"candidates[{index}].opportunityCost",
                ),
                execution_burden=_unit(
                    item["executionBurden"],
                    f"candidates[{index}].executionBurden",
                ),
                equivalence_key=str(item["equivalenceKey"]),
                effects=tuple(cast(Sequence[str], item["effects"])),
                resource_required=round(
                    _unit(
                        snapshot["resourceAvailabilityNormalized"],
                        "resourceAvailabilityNormalized",
                    )
                    * 1000
                )
                if snapshot["task"] == "ITEMIZATION_DECISION"
                else None,
            )
            for index, item in enumerate(candidates_raw)
        )
        required_fields = frozenset(
            {"team", "threats", "candidates", "evidence"}
        )
        context = ScenarioContext(
            observed_at_game_second=int(
                context_raw["observedAtGameSecond"]
            ),
            freshness_seconds=int(context_raw["freshnessSeconds"]),
            completeness=_unit(
                context_raw["completeness"], "context.completeness"
            ),
            uncertain_fields=tuple(
                cast(Sequence[str], context_raw["uncertainFields"])
            ),
            required_fields=required_fields,
            available_fields=required_fields,
            state_signature=str(context_raw["stateSignature"]),
        )
        team_plan = TeamPlan(
            primary_win_condition=str(team_raw["primaryWinCondition"]),
            win_condition_tags=frozenset(
                cast(Sequence[str], team_raw["allyArchetypes"])
            ),
            missing_functions=frozenset(
                cast(Sequence[str], team_raw["missingFunctions"])
            ),
            covered_functions=frozenset(
                cast(Sequence[str], team_raw["allyFunctions"])
            ),
        )
        recent = tuple(
            RecentAdvice(
                action_id=str(item["actionId"]),
                equivalence_key=str(item["equivalenceKey"]),
                age_seconds=int(item["ageSeconds"]),
                decision=str(item["decision"]),
                state_signature=str(item["stateSignature"]),
                category=str(item["category"]),
                reason_codes=tuple(
                    cast(Sequence[str], item["reasonCodes"])
                ),
            )
            for item in recent_raw
        )
        snapshot_id = str(snapshot["snapshotId"])
        task = str(snapshot["task"])
        scenario = CanonicalScenario(
            scenario_id=f"realistic_{hashlib.sha256(snapshot_id.encode()).hexdigest()[:20]}",
            family_id=f"realistic_{task.lower()}",
            split_group=f"realistic_{task.lower()}",
            source_type="REALISTIC_ABSTRACTED",
            seed=_opaque_seed(snapshot_id),
            task=task,
            context=context,
            evidence=evidence,
            candidates=candidates,
            threats=tuple(threats),
            team_plan=team_plan,
            recent_advice=recent,
            output_locale=str(snapshot["outputLocale"]),
            causal_signature=_signature(
                task, context, candidates, threats, team_plan
            ),
        )
        result = self._oracle.decide(scenario)
        provenance = cast(Mapping[str, object], snapshot["provenance"])
        return AdaptedScenario(
            scenario=scenario,
            oracle_result=result,
            review_status=REVIEW_STATUS,
            provenance=MappingProxyType(dict(provenance)),
        )


def adapt_snapshot(snapshot: Mapping[str, object]) -> AdaptedScenario:
    return RealisticAdapter().adapt(snapshot)
