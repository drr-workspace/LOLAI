from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from generators.domain_models import (
    CandidateAction,
    CanonicalScenario,
    Evidence,
)
from generators.oracle import StrategicOracle

from generators.counterfactuals.pairs import EXPECTED_EFFECTS, CounterfactualPair, classify_effect


class CounterfactualError(ValueError):
    pass


class RuleNotApplicableError(CounterfactualError):
    pass


class EffectMismatchError(CounterfactualError):
    pass


@dataclass(frozen=True, slots=True)
class CounterfactualRule:
    id: str
    description: str
    causal_dimension: str
    expected_effect: str


def load_rules(path: Path | None = None) -> MappingProxyType[str, CounterfactualRule]:
    rules_path = path or Path(__file__).with_name("rules.json")
    with rules_path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    parsed: dict[str, CounterfactualRule] = {}
    for raw in document["rules"]:
        rule = CounterfactualRule(
            id=str(raw["id"]),
            description=str(raw["description"]),
            causal_dimension=str(raw["causalDimension"]),
            expected_effect=str(raw["expectedEffect"]),
        )
        if rule.id in parsed:
            raise CounterfactualError(f"regola duplicata: {rule.id}")
        if rule.expected_effect not in EXPECTED_EFFECTS:
            raise CounterfactualError(
                f"expectedEffect non valido: {rule.expected_effect}"
            )
        parsed[rule.id] = rule
    return MappingProxyType(parsed)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuleNotApplicableError(message)


def _selected(
    scenario: CanonicalScenario, selected_id: str | None
) -> CandidateAction:
    _require(selected_id is not None, "la regola richiede un vincitore")
    return next(
        item for item in scenario.candidates if item.action_id == selected_id
    )


def _replace_candidate(
    scenario: CanonicalScenario,
    action_id: str,
    transform: Callable[[CandidateAction], CandidateAction],
) -> CanonicalScenario:
    return replace(
        scenario,
        candidates=tuple(
            transform(item) if item.action_id == action_id else item
            for item in scenario.candidates
        ),
    )


def _mutate(
    scenario: CanonicalScenario,
    rule_id: str,
    oracle: StrategicOracle,
) -> CanonicalScenario:
    parent_result = oracle.decide(scenario)
    if rule_id == "SWAP_PRIMARY_SECONDARY_THREAT":
        _require(len(scenario.threats) >= 2, "servono almeno due minacce")
        ranked = sorted(scenario.threats, key=lambda item: -item.priority)
        first, second = ranked[:2]
        return replace(
            scenario,
            threats=tuple(
                replace(item, priority=second.priority)
                if item.entity_id == first.entity_id
                else replace(item, priority=first.priority)
                if item.entity_id == second.entity_id
                else item
                for item in scenario.threats
            ),
        )
    if rule_id == "REDUCE_EVIDENCE_FRESHNESS":
        return replace(
            scenario,
            evidence=tuple(
                replace(item, freshness_seconds=31)
                for item in scenario.evidence
            ),
        )
    if rule_id == "INTRODUCE_CONTRADICTION":
        _require(len(scenario.evidence) >= 3, "servono tre evidenze")
        ids = [item.evidence_id for item in scenario.evidence[:3]]
        replacements = {
            ids[0]: frozenset({ids[1]}),
            ids[1]: frozenset({ids[0], ids[2]}),
            ids[2]: frozenset({ids[1]}),
        }
        return replace(
            scenario,
            evidence=tuple(
                replace(
                    item,
                    conflicts_with_evidence_ids=replacements.get(
                        item.evidence_id, item.conflicts_with_evidence_ids
                    ),
                )
                for item in scenario.evidence
            ),
        )
    if rule_id == "REMOVE_REQUIRED_EVIDENCE":
        _require(bool(scenario.evidence), "nessuna evidenza da rimuovere")
        return replace(scenario, evidence=())
    if rule_id == "MAKE_WINNER_INFEASIBLE":
        winner = _selected(
            scenario, parent_result.decision.primary_action_id
        )
        _require(len(scenario.candidates) >= 2, "serve un'alternativa")
        return _replace_candidate(
            scenario,
            winner.action_id,
            lambda item: replace(item, feasibility=0.0),
        )
    if rule_id == "MAKE_LOSER_IMMEDIATELY_FEASIBLE":
        winner = _selected(
            scenario, parent_result.decision.primary_action_id
        )
        losers = [
            item
            for item in scenario.candidates
            if item.action_id != winner.action_id
        ]
        _require(bool(losers), "serve un candidato perdente")
        loser = max(
            losers,
            key=lambda item: (
                len(item.countered_threat_ids),
                len(item.supports_functions),
                item.urgency_alignment,
            ),
        )
        return _replace_candidate(
            scenario,
            loser.action_id,
            lambda item: replace(item, feasibility=1.0),
        )
    if rule_id == "FILL_MISSING_TEAM_FUNCTION":
        _require(
            bool(scenario.team_plan.missing_functions),
            "nessuna funzione mancante",
        )
        function = sorted(scenario.team_plan.missing_functions)[0]
        return replace(
            scenario,
            team_plan=replace(
                scenario.team_plan,
                missing_functions=scenario.team_plan.missing_functions
                - {function},
                covered_functions=scenario.team_plan.covered_functions
                | {function},
            ),
        )
    if rule_id == "MAKE_CANDIDATES_EQUIVALENT":
        winner = _selected(
            scenario, parent_result.decision.primary_action_id
        )
        _require(len(scenario.candidates) >= 2, "servono due candidati")
        return replace(
            scenario,
            candidates=tuple(
                replace(
                    item,
                    action_type=winner.action_type,
                    evidence_ids=winner.evidence_ids,
                    feasibility=winner.feasibility,
                    supports_functions=winner.supports_functions,
                    countered_threat_ids=winner.countered_threat_ids,
                    win_condition_tags=winner.win_condition_tags,
                    urgency_alignment=winner.urgency_alignment,
                    opportunity_cost=winner.opportunity_cost,
                    execution_burden=winner.execution_burden,
                    equivalence_key=winner.equivalence_key,
                )
                for item in scenario.candidates
            ),
        )
    if rule_id == "ADD_MEANINGFUL_STATE_CHANGE":
        _require(bool(scenario.recent_advice), "serve advice recente")
        return replace(
            scenario,
            context=replace(
                scenario.context,
                state_signature=f"{scenario.context.state_signature}:changed",
            ),
        )
    if rule_id == "REMOVE_MEANINGFUL_STATE_CHANGE":
        _require(bool(scenario.recent_advice), "serve advice recente")
        return replace(
            scenario,
            context=replace(
                scenario.context,
                state_signature=scenario.recent_advice[0].state_signature,
            ),
        )
    if rule_id == "INCREASE_OBJECTIVE_URGENCY":
        winner = _selected(
            scenario, parent_result.decision.primary_action_id
        )
        return _replace_candidate(
            scenario,
            winner.action_id,
            lambda item: replace(item, urgency_alignment=1.0),
        )
    if rule_id == "RESOLVE_VISION_GAP":
        return replace(
            scenario,
            evidence=tuple(
                replace(item, fact={"visionGapResolved": True})
                if index == 0
                else item
                for index, item in enumerate(scenario.evidence)
            ),
        )
    if rule_id == "CHANGE_WAVE_STATE":
        _require(
            not scenario.recent_advice,
            "wave state con advice recente potrebbe essere rilevante",
        )
        return replace(
            scenario,
            context=replace(
                scenario.context,
                state_signature=f"{scenario.context.state_signature}:wave",
            ),
        )
    if rule_id == "AGE_RECENT_ADVICE":
        _require(bool(scenario.recent_advice), "serve advice recente")
        return replace(
            scenario,
            recent_advice=tuple(
                replace(item, age_seconds=21)
                for item in scenario.recent_advice
            ),
        )
    if rule_id == "ADD_NEW_RELIABLE_EVIDENCE":
        return replace(
            scenario,
            evidence=tuple(
                replace(item, confidence=0.95)
                for item in scenario.evidence
            ),
        )
    raise CounterfactualError(f"regola non implementata: {rule_id}")


def apply_counterfactual(
    scenario: CanonicalScenario,
    rule_id: str,
    *,
    oracle: StrategicOracle | None = None,
    verify: bool = True,
) -> CounterfactualPair:
    rules = load_rules()
    try:
        rule = rules[rule_id]
    except KeyError as error:
        raise CounterfactualError(f"regola sconosciuta: {rule_id}") from error
    engine = oracle or StrategicOracle()
    parent_result = engine.decide(scenario)
    mutated = _mutate(scenario, rule_id, engine)
    pair_material = f"{scenario.scenario_id}:{rule_id}".encode("utf-8")
    pair_id = f"pair_{hashlib.sha256(pair_material).hexdigest()[:16]}"
    child = replace(
        mutated,
        scenario_id=f"{scenario.scenario_id}_cf_{rule_id.lower()}",
        source_type="COUNTERFACTUAL",
        parent_scenario_id=scenario.scenario_id,
        counterfactual_pair_id=pair_id,
    )
    child_result = engine.decide(child)
    observed = classify_effect(parent_result, child_result)
    if verify and observed != rule.expected_effect:
        raise EffectMismatchError(
            f"{rule_id}: effetto {observed}, atteso {rule.expected_effect}"
        )
    return CounterfactualPair(
        rule_id=rule.id,
        expected_effect=rule.expected_effect,
        observed_effect=observed,
        parent=scenario,
        counterfactual=child,
        parent_result=parent_result,
        counterfactual_result=child_result,
    )
