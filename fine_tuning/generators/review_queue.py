from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from generators.realistic_adapter import AdaptedScenario
from generators.scoring import contradiction_count


class ReviewQueueError(ValueError):
    pass


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _decision(record: AdaptedScenario) -> dict[str, object]:
    value = record.oracle_result.decision
    return {
        "schemaVersion": value.schema_version,
        "decision": value.decision,
        "category": value.category,
        "primaryActionId": value.primary_action_id,
        "alternativeActionIds": list(value.alternative_action_ids),
        "priority": value.priority,
        "confidence": value.confidence,
        "reasonCodes": list(value.reason_codes),
        "evidenceIds": list(value.evidence_ids),
        "message": "",
        "validForSeconds": value.valid_for_seconds,
        "recheckTriggers": list(value.recheck_triggers),
    }


def _scenario(record: AdaptedScenario) -> dict[str, object]:
    value = record.scenario
    return {
        "scenarioId": value.scenario_id,
        "familyId": value.family_id,
        "splitGroup": value.split_group,
        "sourceType": value.source_type,
        "seed": value.seed,
        "parentScenarioId": value.parent_scenario_id,
        "counterfactualPairId": value.counterfactual_pair_id,
        "episodeId": value.episode_id,
        "episodeStep": value.episode_step,
        "causalSignature": value.causal_signature,
        "task": value.task,
        "outputLocale": value.output_locale,
        "context": {
            "observedAtGameSecond": value.context.observed_at_game_second,
            "freshnessSeconds": value.context.freshness_seconds,
            "completeness": value.context.completeness,
            "uncertainFields": list(value.context.uncertain_fields),
            "requiredFields": sorted(value.context.required_fields),
            "availableFields": sorted(value.context.available_fields),
            "stateSignature": value.context.state_signature,
        },
        "evidence": [
            {
                "evidenceId": item.evidence_id,
                "type": item.category,
                "confidence": item.confidence,
                "freshnessSeconds": item.freshness_seconds,
                "conflictsWithEvidenceIds": sorted(
                    item.conflicts_with_evidence_ids
                ),
                "fact": item.fact,
            }
            for item in value.evidence
        ],
        "threats": [
            {
                "entityId": item.entity_id,
                "priority": item.priority,
                "evidenceIds": list(item.evidence_ids),
                "patterns": list(item.patterns),
                "damageProfile": {
                    "physical": item.damage_profile[0],
                    "magic": item.damage_profile[1],
                    "true": item.damage_profile[2],
                },
            }
            for item in value.threats
        ],
        "candidates": [
            {
                "actionId": item.action_id,
                "type": item.action_type,
                "evidenceIds": list(item.evidence_ids),
                "feasibility": item.feasibility,
                "supportsFunctions": sorted(item.supports_functions),
                "counteredThreatIds": sorted(item.countered_threat_ids),
                "winConditionTags": sorted(item.win_condition_tags),
                "urgencyAlignment": item.urgency_alignment,
                "opportunityCost": item.opportunity_cost,
                "executionBurden": item.execution_burden,
                "equivalenceKey": item.equivalence_key,
                "effects": list(item.effects),
                "resourceRequired": item.resource_required,
            }
            for item in value.candidates
        ],
        "teamPlan": {
            "primaryWinCondition": value.team_plan.primary_win_condition,
            "winConditionTags": sorted(value.team_plan.win_condition_tags),
            "missingFunctions": sorted(value.team_plan.missing_functions),
            "coveredFunctions": sorted(value.team_plan.covered_functions),
        },
        "recentAdvice": [
            {
                "actionId": item.action_id,
                "equivalenceKey": item.equivalence_key,
                "ageSeconds": item.age_seconds,
                "decision": item.decision,
                "stateSignature": item.state_signature,
                "category": item.category,
                "reasonCodes": list(item.reason_codes),
            }
            for item in value.recent_advice
        ],
    }


def _priority(record: AdaptedScenario, novel_signature: bool) -> float:
    decision = record.oracle_result.decision
    trace = record.oracle_result.trace
    low_confidence = (1.0 - decision.confidence) * 50.0
    conflicts = min(20.0, contradiction_count(record.scenario) * 10.0)
    borderline = (
        20.0
        if decision.decision == "SHOW" and trace.score_margin <= 0.1
        else 0.0
    )
    novelty = 10.0 if novel_signature else 0.0
    return round(low_confidence + conflicts + borderline + novelty, 3)


class ReviewQueue:
    def __init__(self, queue_path: Path, audit_path: Path) -> None:
        self.queue_path = queue_path
        self.audit_path = audit_path

    def _read(self) -> list[dict[str, Any]]:
        if not self.queue_path.exists():
            return []
        entries: list[dict[str, Any]] = []
        with self.queue_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ReviewQueueError(
                        f"{self.queue_path}:{line_number}: oggetto atteso"
                    )
                entries.append(value)
        return entries

    def _write(self, entries: Iterable[Mapping[str, object]]) -> None:
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.queue_path.with_suffix(
            f"{self.queue_path.suffix}.tmp"
        )
        with temporary.open("w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(_canonical_json(entry))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.queue_path)

    def _audit(
        self,
        *,
        scenario_id: str,
        action: str,
        reviewer: str,
        before: object,
        after: object,
        details: Mapping[str, object] | None = None,
    ) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "timestamp": _now(),
            "scenarioId": scenario_id,
            "action": action,
            "reviewer": reviewer,
            "beforeHash": hashlib.sha256(
                _canonical_json(before).encode("utf-8")
            ).hexdigest(),
            "afterHash": hashlib.sha256(
                _canonical_json(after).encode("utf-8")
            ).hexdigest(),
            "details": dict(details or {}),
        }
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(_canonical_json(event))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def enqueue(self, record: AdaptedScenario) -> bool:
        entries = self._read()
        scenario_id = record.scenario.scenario_id
        signature = record.scenario.causal_signature
        if any(
            entry.get("scenario", {}).get("scenarioId") == scenario_id
            or entry.get("scenario", {}).get("causalSignature") == signature
            for entry in entries
        ):
            return False
        known_signatures = {
            str(entry.get("scenario", {}).get("causalSignature"))
            for entry in entries
        }
        entry: dict[str, object] = {
            "scenario": _scenario(record),
            "expectedOutput": _decision(record),
            "reviewStatus": record.review_status,
            "reviewPriority": _priority(
                record, signature not in known_signatures
            ),
            "provenance": dict(record.provenance),
            "notes": [],
            "createdAt": _now(),
            "updatedAt": _now(),
        }
        entries.append(entry)
        entries.sort(
            key=lambda item: (
                -float(item["reviewPriority"]),
                str(item["scenario"]["scenarioId"]),
            )
        )
        self._write(entries)
        return True

    def list(
        self, status: str | None = None
    ) -> tuple[Mapping[str, object], ...]:
        entries = self._read()
        if status is not None:
            entries = [
                entry
                for entry in entries
                if entry.get("reviewStatus") == status
            ]
        return tuple(entries)

    def show(self, scenario_id: str) -> Mapping[str, object]:
        for entry in self._read():
            if entry["scenario"]["scenarioId"] == scenario_id:
                return entry
        raise ReviewQueueError(f"scenarioId sconosciuto: {scenario_id}")

    def _mutate(
        self,
        scenario_id: str,
        action: str,
        reviewer: str,
        mutate: Any,
        details: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        entries = self._read()
        for index, entry in enumerate(entries):
            if entry["scenario"]["scenarioId"] != scenario_id:
                continue
            before = json.loads(json.dumps(entry))
            mutate(entry)
            entry["updatedAt"] = _now()
            entries[index] = entry
            self._write(entries)
            self._audit(
                scenario_id=scenario_id,
                action=action,
                reviewer=reviewer,
                before=before,
                after=entry,
                details=details,
            )
            return entry
        raise ReviewQueueError(f"scenarioId sconosciuto: {scenario_id}")

    def set_status(
        self, scenario_id: str, status: str, reviewer: str
    ) -> Mapping[str, object]:
        if status not in {"APPROVED", "REJECTED"}:
            raise ReviewQueueError(f"status non valido: {status}")
        return self._mutate(
            scenario_id,
            status.lower(),
            reviewer,
            lambda entry: entry.__setitem__("reviewStatus", status),
        )

    def edit_expected(
        self,
        scenario_id: str,
        field: str,
        value: object,
        reviewer: str,
    ) -> Mapping[str, object]:
        allowed = {
            "decision",
            "primaryActionId",
            "alternativeActionIds",
            "priority",
            "confidence",
            "reasonCodes",
            "evidenceIds",
            "message",
            "validForSeconds",
            "recheckTriggers",
        }
        if field not in allowed:
            raise ReviewQueueError(
                f"campo expectedOutput non modificabile: {field}"
            )
        return self._mutate(
            scenario_id,
            "edit-expected",
            reviewer,
            lambda entry: entry["expectedOutput"].__setitem__(field, value),
            {"field": field},
        )

    def add_note(
        self, scenario_id: str, note: str, reviewer: str
    ) -> Mapping[str, object]:
        if not note.strip():
            raise ReviewQueueError("nota vuota")
        return self._mutate(
            scenario_id,
            "add-note",
            reviewer,
            lambda entry: entry["notes"].append(
                {
                    "reviewer": reviewer,
                    "note": note,
                    "timestamp": _now(),
                }
            ),
        )

    def export_approved(self, output_path: Path) -> int:
        approved = [
            entry
            for entry in self._read()
            if entry.get("reviewStatus") == "APPROVED"
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for entry in approved:
                handle.write(_canonical_json(entry))
                handle.write("\n")
        return len(approved)
