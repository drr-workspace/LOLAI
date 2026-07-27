from __future__ import annotations

from collections.abc import Iterable, Mapping

from generators.domain_models import (
    CandidateAction,
    CandidateScore,
    CanonicalScenario,
    ConfidenceContribution,
    OracleDecision,
    OracleResult,
    OracleTrace,
)
from generators.ontology_registry import (
    OntologyRegistry,
    load_ontology,
)
from generators.policy_loader import PolicyBundle, load_policies
from generators.scoring import (
    contradiction_count,
    score_candidates,
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _candidate_by_id(
    scenario: CanonicalScenario,
) -> Mapping[str, CandidateAction]:
    candidates = {candidate.action_id: candidate for candidate in scenario.candidates}
    if len(candidates) != len(scenario.candidates):
        raise ValueError("gli actionId dei candidati devono essere univoci")
    return candidates


def _semantic_sort_key(candidate: CandidateAction) -> tuple[object, ...]:
    return (
        candidate.action_type,
        tuple(sorted(candidate.supports_functions)),
        tuple(sorted(candidate.countered_threat_ids)),
        tuple(sorted(candidate.win_condition_tags)),
        candidate.feasibility,
        candidate.urgency_alignment,
        candidate.opportunity_cost,
        candidate.execution_burden,
        candidate.equivalence_key,
    )


def _known_evidence_ids(
    scenario: CanonicalScenario,
    requested_ids: Iterable[str],
) -> tuple[str, ...]:
    known = {evidence.evidence_id for evidence in scenario.evidence}
    return tuple(sorted(set(requested_ids) & known))


class StrategicOracle:
    """Deterministic policy-driven oracle with no model or name-based inference."""

    def __init__(
        self,
        policies: PolicyBundle | None = None,
        ontology: OntologyRegistry | None = None,
    ) -> None:
        self._policies = policies or load_policies()
        self._ontology = ontology or load_ontology()
        if self._policies.ontology_version != self._ontology.version:
            raise ValueError("versione ontologia incoerente tra oracle e policy")

    def decide(self, scenario: CanonicalScenario) -> OracleResult:
        """Apply refresh, scoring, repetition and delivery rules in order."""
        if not self._ontology.contains("tasks", scenario.task):
            raise ValueError(f"task non valido: {scenario.task}")
        candidates_by_id = _candidate_by_id(scenario)
        contradiction_total = contradiction_count(scenario)
        gates: list[str] = []

        refresh_gate, refresh_reasons, refresh_evidence = self._refresh_gate(
            scenario, contradiction_total, gates
        )
        if refresh_gate is not None:
            return self._build_result(
                scenario=scenario,
                decision="REQUEST_REFRESH",
                selected=None,
                alternatives=(),
                scores=(),
                margin=0.0,
                gates=gates,
                triggered_gate=refresh_gate,
                contradiction_total=contradiction_total,
                repeated_key=None,
                preferred_reasons=refresh_reasons,
                evidence_ids=refresh_evidence,
            )

        gates.append("scoreCandidates")
        scores = score_candidates(
            scenario,
            self._policies.scoring,
            self._policies.decision.thresholds.stale_context_seconds,
        )
        valid_scores = [score for score in scores if score.valid]
        valid_scores.sort(
            key=lambda score: (
                -score.total,
                _semantic_sort_key(candidates_by_id[score.action_id]),
                score.action_id,
            )
        )

        if not valid_scores:
            gates.append("noValidCandidate")
            return self._build_result(
                scenario=scenario,
                decision="SUPPRESS",
                selected=None,
                alternatives=(),
                scores=scores,
                margin=0.0,
                gates=gates,
                triggered_gate="noValidCandidate",
                contradiction_total=contradiction_total,
                repeated_key=None,
                preferred_reasons=("NO_MEANINGFUL_ADVANTAGE", "LOW_ACTION_VALUE"),
                evidence_ids=(),
            )

        best_score = valid_scores[0]
        best_candidate = candidates_by_id[best_score.action_id]
        second_score = valid_scores[1].total if len(valid_scores) > 1 else 0.0
        margin = max(0.0, best_score.total - second_score)

        gates.append("recentEquivalentAdvice")
        repeated_key = self._recent_equivalent_key(scenario, best_candidate)
        if repeated_key is not None:
            return self._build_result(
                scenario=scenario,
                decision="SUPPRESS",
                selected=None,
                alternatives=(),
                scores=scores,
                margin=margin,
                gates=gates,
                triggered_gate="recentEquivalentAdvice",
                contradiction_total=contradiction_total,
                repeated_key=repeated_key,
                preferred_reasons=("RECENTLY_ADVISED", "NO_NEW_INFORMATION"),
                evidence_ids=_known_evidence_ids(
                    scenario, best_candidate.evidence_ids
                ),
            )

        gates.append("meaningfulScoreMargin")
        if (
            len(valid_scores) > 1
            and margin
            < self._policies.decision.thresholds.meaningful_score_margin
        ):
            return self._build_result(
                scenario=scenario,
                decision="SUPPRESS",
                selected=None,
                alternatives=(),
                scores=scores,
                margin=margin,
                gates=gates,
                triggered_gate="insufficientScoreMargin",
                contradiction_total=contradiction_total,
                repeated_key=None,
                preferred_reasons=(
                    "NO_MEANINGFUL_ADVANTAGE",
                    "LOW_ACTION_VALUE",
                ),
                evidence_ids=_known_evidence_ids(
                    scenario,
                    (
                        *best_candidate.evidence_ids,
                        *candidates_by_id[valid_scores[1].action_id].evidence_ids,
                    ),
                ),
            )

        gates.append("lowActionValue")
        if (
            best_score.total
            < self._policies.decision.thresholds.low_action_value_threshold
        ):
            return self._build_result(
                scenario=scenario,
                decision="SUPPRESS",
                selected=None,
                alternatives=(),
                scores=scores,
                margin=margin,
                gates=gates,
                triggered_gate="lowActionValue",
                contradiction_total=contradiction_total,
                repeated_key=None,
                preferred_reasons=("LOW_ACTION_VALUE",),
                evidence_ids=_known_evidence_ids(
                    scenario, best_candidate.evidence_ids
                ),
            )

        gates.append("showCandidate")
        scoring_reasons = self._scoring_reason_codes(best_score)
        alternatives = tuple(score.action_id for score in valid_scores[1:])
        return self._build_result(
            scenario=scenario,
            decision="SHOW",
            selected=best_candidate,
            alternatives=alternatives,
            scores=scores,
            margin=margin,
            gates=gates,
            triggered_gate="showCandidate",
            contradiction_total=contradiction_total,
            repeated_key=None,
            preferred_reasons=scoring_reasons,
            evidence_ids=_known_evidence_ids(
                scenario, best_candidate.evidence_ids
            ),
        )

    def _refresh_gate(
        self,
        scenario: CanonicalScenario,
        contradiction_total: int,
        gates: list[str],
    ) -> tuple[str | None, tuple[str, ...], tuple[str, ...]]:
        thresholds = self._policies.decision.thresholds
        gates.append("staleRequiredContext")
        stale_evidence = tuple(
            evidence.evidence_id
            for evidence in scenario.evidence
            if evidence.freshness_seconds > thresholds.stale_context_seconds
        )
        if (
            scenario.context.freshness_seconds > thresholds.stale_context_seconds
            or stale_evidence
        ):
            return "staleRequiredContext", ("STALE_CONTEXT",), tuple(
                sorted(stale_evidence)
            )

        gates.append("incompleteContext")
        if (
            scenario.context.completeness < thresholds.minimum_completeness
            or scenario.context.missing_required_fields
        ):
            return (
                "incompleteContext",
                ("INSUFFICIENT_CONTEXT",),
                tuple(evidence.evidence_id for evidence in scenario.evidence),
            )

        gates.append("contradictoryEvidence")
        if contradiction_total > thresholds.maximum_contradiction_count:
            conflicting = tuple(
                evidence.evidence_id
                for evidence in scenario.evidence
                if evidence.conflicts_with_evidence_ids
            )
            return (
                "contradictoryEvidence",
                ("CONFLICTING_EVIDENCE",),
                tuple(sorted(conflicting)),
            )

        gates.append("unreliableEvidence")
        evidence_quality = self._evidence_quality(scenario, None)
        if evidence_quality < thresholds.minimum_evidence_confidence:
            unreliable = tuple(
                evidence.evidence_id
                for evidence in scenario.evidence
                if evidence.confidence < thresholds.minimum_evidence_confidence
            )
            return (
                "unreliableEvidence",
                ("INSUFFICIENT_CONTEXT",),
                tuple(sorted(unreliable)),
            )
        return None, (), ()

    def _recent_equivalent_key(
        self,
        scenario: CanonicalScenario,
        candidate: CandidateAction,
    ) -> str | None:
        if not candidate.equivalence_key:
            return None
        threshold = self._policies.decision.thresholds.recently_advised_seconds
        for advice in scenario.recent_advice:
            if (
                advice.decision == "SHOW"
                and advice.equivalence_key == candidate.equivalence_key
                and advice.age_seconds <= threshold
                and (
                    not advice.state_signature
                    or advice.state_signature == scenario.context.state_signature
                )
            ):
                return candidate.equivalence_key
        return None

    def _scoring_reason_codes(
        self,
        score: CandidateScore,
    ) -> tuple[str, ...]:
        component_by_id = {
            component.id: component
            for component in self._policies.scoring.components
        }
        reasons: list[str] = []
        for contribution in sorted(
            score.contributions,
            key=lambda item: (-item.contribution, item.component_id),
        ):
            if contribution.contribution <= 0.0:
                continue
            for reason in component_by_id[contribution.component_id].reason_codes:
                if reason not in reasons:
                    reasons.append(reason)
        return tuple(reasons)

    def _compatible_reasons(
        self,
        preferred: Iterable[str],
        task: str,
        decision: str,
    ) -> tuple[str, ...]:
        compatible: list[str] = []
        for reason in preferred:
            if not self._ontology.contains("reason-codes", reason):
                continue
            entry = self._ontology.require("reason-codes", reason)
            if (
                task in entry.get("allowedTasks", ())
                and decision in entry.get("decisionCompatibility", ())
                and reason not in compatible
            ):
                compatible.append(reason)
        if compatible:
            return tuple(compatible[:3])

        fallback = sorted(
            str(entry["id"])
            for entry in self._ontology.values("reason-codes")
            if task in entry.get("allowedTasks", ())
            and decision in entry.get("decisionCompatibility", ())
        )
        if not fallback:
            raise ValueError(
                f"nessun reason code compatibile con {task}/{decision}"
            )
        return (fallback[0],)

    def _evidence_quality(
        self,
        scenario: CanonicalScenario,
        selected: CandidateAction | None,
    ) -> float:
        selected_ids = set(selected.evidence_ids) if selected else set()
        relevant = [
            evidence
            for evidence in scenario.evidence
            if not selected_ids or evidence.evidence_id in selected_ids
        ]
        if not relevant:
            return 0.0
        return _clamp(
            sum(evidence.confidence for evidence in relevant) / len(relevant)
        )

    def _confidence(
        self,
        scenario: CanonicalScenario,
        selected: CandidateAction | None,
        margin: float,
        contradiction_total: int,
    ) -> tuple[float, float, tuple[ConfidenceContribution, ...]]:
        thresholds = self._policies.decision.thresholds
        relevant_ids = set(selected.evidence_ids) if selected else set()
        relevant_evidence = [
            evidence
            for evidence in scenario.evidence
            if not relevant_ids or evidence.evidence_id in relevant_ids
        ]
        average_freshness = (
            sum(item.freshness_seconds for item in relevant_evidence)
            / len(relevant_evidence)
            if relevant_evidence
            else thresholds.stale_context_seconds
        )
        freshness_value = 1.0 - _clamp(
            average_freshness / thresholds.stale_context_seconds
        )
        uncertain_denominator = max(
            len(scenario.context.required_fields),
            len(scenario.context.uncertain_fields),
            1,
        )
        raw_values = {
            "evidenceQuality": self._evidence_quality(scenario, selected),
            "evidenceFreshness": freshness_value,
            "contextCompleteness": _clamp(scenario.context.completeness),
            "candidateScoreMargin": _clamp(margin),
            "conflictCount": _clamp(
                contradiction_total
                / max(thresholds.maximum_contradiction_count + 1, 1)
            ),
            "uncertainFieldCount": _clamp(
                len(scenario.context.uncertain_fields)
                / uncertain_denominator
            ),
        }
        contributions: list[ConfidenceContribution] = []
        for factor in self._policies.confidence.factors:
            raw_value = raw_values[factor.id]
            effective_value = (
                raw_value if factor.direction == "POSITIVE" else 1.0 - raw_value
            )
            contributions.append(
                ConfidenceContribution(
                    factor_id=factor.id,
                    raw_value=raw_value,
                    direction=factor.direction,
                    weight=factor.weight,
                    contribution=factor.weight * effective_value,
                )
            )
        unrounded = sum(item.contribution for item in contributions)
        bounded = max(
            self._policies.confidence.output_range.minimum,
            min(self._policies.confidence.output_range.maximum, unrounded),
        )
        confidence = round(
            bounded, self._policies.confidence.rounding_digits
        )
        return confidence, unrounded, tuple(contributions)

    def _build_result(
        self,
        scenario: CanonicalScenario,
        decision: str,
        selected: CandidateAction | None,
        alternatives: tuple[str, ...],
        scores: tuple[CandidateScore, ...],
        margin: float,
        gates: list[str],
        triggered_gate: str,
        contradiction_total: int,
        repeated_key: str | None,
        preferred_reasons: Iterable[str],
        evidence_ids: tuple[str, ...],
    ) -> OracleResult:
        reasons = self._compatible_reasons(
            preferred_reasons, scenario.task, decision
        )
        confidence, confidence_unrounded, confidence_contributions = (
            self._confidence(
                scenario,
                selected,
                margin,
                contradiction_total,
            )
        )
        ranked_ids = tuple(
            score.action_id
            for score in sorted(
                (score for score in scores if score.valid),
                key=lambda score: (-score.total, score.action_id),
            )
        )
        if decision == "SHOW":
            priority = "HIGH" if selected and any(
                score.action_id == selected.action_id and score.total >= 0.6
                for score in scores
            ) else "MEDIUM"
            valid_for = 30
            recheck = ("ALLY_OR_ENEMY_STATE_CHANGED", "NEW_RELIABLE_EVIDENCE")
        elif decision == "REQUEST_REFRESH":
            priority = "LOW"
            valid_for = 3
            recheck = ("CONTEXT_REFRESHED", "NEW_RELIABLE_EVIDENCE")
        else:
            priority = "LOW"
            valid_for = 10
            recheck = ("NEW_RELIABLE_EVIDENCE", "TIME_WINDOW_EXPIRED")

        oracle_decision = OracleDecision(
            schema_version="1.0.0",
            decision=decision,
            category=scenario.task,
            primary_action_id=selected.action_id if selected else None,
            alternative_action_ids=alternatives if selected else (),
            priority=priority,
            confidence=confidence,
            reason_codes=reasons,
            evidence_ids=tuple(sorted(set(evidence_ids))),
            valid_for_seconds=valid_for,
            recheck_triggers=recheck,
        )
        trace = OracleTrace(
            scenario_id=scenario.scenario_id,
            seed=scenario.seed,
            gates_evaluated=tuple(gates),
            triggered_gate=triggered_gate,
            contradiction_count=contradiction_total,
            missing_required_fields=tuple(
                sorted(scenario.context.missing_required_fields)
            ),
            candidate_scores=scores,
            ranked_action_ids=ranked_ids,
            score_margin=margin,
            repeated_equivalence_key=repeated_key,
            confidence_contributions=confidence_contributions,
            confidence_unrounded=confidence_unrounded,
            final_decision=decision,
            selected_action_id=selected.action_id if selected else None,
            reason_codes=reasons,
            evidence_ids=tuple(sorted(set(evidence_ids))),
        )
        return OracleResult(decision=oracle_decision, trace=trace)


def run_oracle(
    scenario: CanonicalScenario,
    policies: PolicyBundle | None = None,
    ontology: OntologyRegistry | None = None,
) -> OracleResult:
    """Convenience function for one deterministic oracle decision."""
    return StrategicOracle(policies=policies, ontology=ontology).decide(scenario)
