from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from generators.domain_models import (
    CandidateAction,
    CanonicalScenario,
    OracleResult,
)
from generators.policy_loader import (
    DeduplicationPolicy,
    load_policies,
)


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    """Scenario etichettato con i soli metadati necessari al dataset."""

    scenario: CanonicalScenario
    oracle_result: OracleResult
    review_status: str = "GENERATED"
    counterfactual_variable: str | None = None


@dataclass(frozen=True, slots=True)
class CausalSignature:
    """Firma leggibile e digest stabile della struttura causale."""

    digest: str
    components: tuple[tuple[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return dict(self.components)


class CausalSignatureBuilder:
    """Costruisce firme invarianti rispetto a ID, ordine e formulazione."""

    def __init__(self, policy: DeduplicationPolicy | None = None) -> None:
        self._policy = policy or load_policies().deduplication

    def build(self, record: DatasetRecord) -> CausalSignature:
        scenario = record.scenario
        result = record.oracle_result
        candidates = tuple(
            sorted(
                (
                    self._candidate_semantics(candidate, scenario)
                    for candidate in scenario.candidates
                ),
                key=_canonical_json,
            )
        )
        selected = next(
            (
                candidate
                for candidate in scenario.candidates
                if candidate.action_id
                == result.decision.primary_action_id
            ),
            None,
        )
        selected_evidence_ids = set(result.decision.evidence_ids)
        components: dict[str, Any] = {
            "task": scenario.task,
            "familyId": scenario.family_id,
            "threatPriorityRelation": self._threat_relation(scenario),
            "candidateFeasibilityRelation": candidates,
            "winnerEffects": self._winner_effects(selected, scenario),
            "missingFunctions": sorted(
                scenario.team_plan.missing_functions
            ),
            "winCondition": {
                "primary": scenario.team_plan.primary_win_condition,
                "tags": sorted(scenario.team_plan.win_condition_tags),
            },
            "contextQuality": {
                "completeness": self._bucket(
                    "input.context.completeness",
                    scenario.context.completeness,
                ),
                "freshness": self._bucket(
                    "input.context.freshnessSeconds",
                    scenario.context.freshness_seconds,
                ),
                "missingRequiredCount": len(
                    scenario.context.missing_required_fields
                ),
                "uncertainCount": len(scenario.context.uncertain_fields),
                "contradictionCount": result.trace.contradiction_count,
            },
            "decision": result.decision.decision,
            "reasonCodes": sorted(result.decision.reason_codes),
            "evidenceCategories": sorted(
                {
                    evidence.category
                    for evidence in scenario.evidence
                    if evidence.evidence_id in selected_evidence_ids
                }
            ),
            "abstentionCause": (
                result.trace.triggered_gate
                if result.decision.decision != "SHOW"
                else None
            ),
            "counterfactualVariable": record.counterfactual_variable,
        }
        canonical = _canonical_json(components)
        return CausalSignature(
            digest=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            components=tuple(sorted(components.items())),
        )

    def structural_features(self, record: DatasetRecord) -> frozenset[str]:
        """Restituisce feature atomiche per la similarità strutturale."""

        signature = self._structural_semantics(record)
        features: set[str] = set()
        _flatten_features("", signature, features)
        semantic_digest = self.structural_key(record)
        # A change in normalized causal semantics must weigh more than a
        # cosmetic leaf difference in the Jaccard feature set. Repeated
        # namespaces act as an explicit weight while preserving set-based
        # similarity and all ID/order invariants.
        for index in range(8):
            features.add(f".causalSemantics[{index}]={semantic_digest}")
        return frozenset(features)

    def structural_key(self, record: DatasetRecord) -> str:
        """Index key for records eligible for near-duplicate comparison."""

        return hashlib.sha256(
            _canonical_json(self._structural_semantics(record)).encode(
                "utf-8"
            )
        ).hexdigest()

    def _structural_semantics(
        self, record: DatasetRecord
    ) -> dict[str, Any]:
        signature = self.build(record).as_dict()
        signature.pop("familyId", None)
        signature.pop("counterfactualVariable", None)
        return signature

    def _bucket(self, name: str, value: float) -> int:
        boundaries = self._policy.numeric_buckets[name]
        for index, upper in enumerate(boundaries[1:]):
            if value <= upper:
                return index
        return len(boundaries) - 1

    def _threat_relation(
        self, scenario: CanonicalScenario
    ) -> tuple[dict[str, Any], ...]:
        ordered = sorted(
            scenario.threats,
            key=lambda threat: (
                -threat.priority,
                threat.patterns,
                threat.damage_profile,
            ),
        )
        return tuple(
            {
                "rank": index,
                "priorityBand": _unit_bucket(threat.priority),
                "patterns": sorted(threat.patterns),
                "damageProfile": tuple(
                    _unit_bucket(value) for value in threat.damage_profile
                ),
            }
            for index, threat in enumerate(ordered)
        )

    def _candidate_semantics(
        self,
        candidate: CandidateAction,
        scenario: CanonicalScenario,
    ) -> dict[str, Any]:
        threat_by_id = {
            threat.entity_id: threat for threat in scenario.threats
        }
        countered = [
            threat_by_id[entity_id]
            for entity_id in candidate.countered_threat_ids
            if entity_id in threat_by_id
        ]
        return {
            "type": candidate.action_type,
            "feasibility": _unit_bucket(candidate.feasibility),
            "functions": sorted(candidate.supports_functions),
            "winConditionTags": sorted(candidate.win_condition_tags),
            "counteredThreatRanks": sorted(
                _priority_rank(threat, scenario) for threat in countered
            ),
            "effects": sorted(candidate.effects),
            "urgency": _unit_bucket(candidate.urgency_alignment),
            "opportunityCost": _unit_bucket(candidate.opportunity_cost),
            "executionBurden": _unit_bucket(candidate.execution_burden),
        }

    def _winner_effects(
        self,
        selected: CandidateAction | None,
        scenario: CanonicalScenario,
    ) -> dict[str, Any] | None:
        if selected is None:
            return None
        return self._candidate_semantics(selected, scenario)


def build_causal_signature(
    scenario: CanonicalScenario,
    oracle_result: OracleResult,
    *,
    counterfactual_variable: str | None = None,
    policy: DeduplicationPolicy | None = None,
) -> CausalSignature:
    return CausalSignatureBuilder(policy).build(
        DatasetRecord(
            scenario=scenario,
            oracle_result=oracle_result,
            counterfactual_variable=counterfactual_variable,
        )
    )


def _priority_rank(threat: object, scenario: CanonicalScenario) -> int:
    priorities = sorted(
        {item.priority for item in scenario.threats}, reverse=True
    )
    return priorities.index(getattr(threat, "priority"))


def _unit_bucket(value: float) -> int:
    return min(4, max(0, int(float(value) * 5)))


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _flatten_features(
    prefix: str, value: object, target: set[str]
) -> None:
    if isinstance(value, dict):
        for key, child in sorted(value.items()):
            _flatten_features(f"{prefix}.{key}", child, target)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _flatten_features(prefix, child, target)
    else:
        target.add(f"{prefix}={value!r}")
