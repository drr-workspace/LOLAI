from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from generators.domain_models import (
    CandidateAction,
    CanonicalScenario,
    Evidence,
    OracleDecision,
    OracleResult,
    RecentAdvice,
    ScenarioContext,
    TeamPlan,
    Threat,
)
from generators.oracle import StrategicOracle


SEQUENCE_TYPES = frozenset(
    {
        "VISION_CONTEST",
        "ITEMIZATION_THREAT_CHANGE",
        "WAVE_POSITION_CHANGE",
        "MACRO_CONTRADICTION",
        "REFRESH_RECOVERY",
        "VALIDITY_EXPIRY",
    }
)


@dataclass(frozen=True, slots=True)
class EpisodeTemplate:
    template_id: str
    description: str
    sequence_type: str
    task: str
    split_group: str
    elapsed_seconds: tuple[int, ...]
    expected_decisions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EpisodeStep:
    index: int
    elapsed_seconds: int
    previous_state: CanonicalScenario | None
    delta: Mapping[str, object]
    scenario: CanonicalScenario
    expected_transition: str
    previous_advice: OracleDecision | None
    oracle_result: OracleResult


@dataclass(frozen=True, slots=True)
class Episode:
    episode_id: str
    template_id: str
    split_name: str
    split_group: str
    steps: tuple[EpisodeStep, ...]


class EpisodeError(ValueError):
    pass


def load_episode_templates(
    path: Path | None = None,
) -> Mapping[str, EpisodeTemplate]:
    template_path = path or Path(__file__).with_name(
        "episode_templates.json"
    )
    with template_path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    templates: dict[str, EpisodeTemplate] = {}
    for raw in document["templates"]:
        template = EpisodeTemplate(
            template_id=str(raw["templateId"]),
            description=str(raw["description"]),
            sequence_type=str(raw["sequenceType"]),
            task=str(raw["task"]),
            split_group=str(raw["splitGroup"]),
            elapsed_seconds=tuple(int(value) for value in raw["elapsedSeconds"]),
            expected_decisions=tuple(
                str(value) for value in raw["expectedDecisions"]
            ),
        )
        if template.template_id in templates:
            raise EpisodeError(
                f"template episodio duplicato: {template.template_id}"
            )
        if template.sequence_type not in SEQUENCE_TYPES:
            raise EpisodeError(
                f"sequenceType non valido: {template.sequence_type}"
            )
        if not 3 <= len(template.elapsed_seconds) <= 8:
            raise EpisodeError(
                f"{template.template_id}: servono da 3 a 8 step"
            )
        if len(template.elapsed_seconds) != len(
            template.expected_decisions
        ):
            raise EpisodeError(
                f"{template.template_id}: tempi e decisioni non allineati"
            )
        templates[template.template_id] = template
    if len(templates) < 20:
        raise EpisodeError("servono almeno 20 template di episodio")
    return MappingProxyType(templates)


def _episode_id(template_id: str, seed: int) -> str:
    digest = hashlib.sha256(
        f"{template_id}:{seed}".encode("utf-8")
    ).hexdigest()[:18]
    return f"episode_{digest}"


def _split_name(episode_id: str) -> str:
    bucket = int(
        hashlib.sha256(episode_id.encode("utf-8")).hexdigest()[:8], 16
    ) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "valid"
    return "test"


def _action_types(sequence_type: str) -> tuple[str, str]:
    return {
        "VISION_CONTEST": ("SETUP_VISION", "CONTEST_OBJECTIVE"),
        "ITEMIZATION_THREAT_CHANGE": ("PURCHASE", "PURCHASE"),
        "WAVE_POSITION_CHANGE": ("POSITION_SAFELY", "RECALL"),
        "MACRO_CONTRADICTION": ("ROTATE", "HOLD"),
        "REFRESH_RECOVERY": ("PEEL", "HOLD"),
        "VALIDITY_EXPIRY": ("GROUP", "HOLD"),
    }[sequence_type]


def _base_scenario(
    template: EpisodeTemplate,
    episode_id: str,
    seed: int,
) -> CanonicalScenario:
    action_types = _action_types(template.sequence_type)
    evidence = tuple(
        Evidence(
            evidence_id=f"{episode_id}_evidence_{index}",
            category="OBSERVED_RUNTIME",
            confidence=0.95,
            freshness_seconds=1,
            fact={"signal": f"temporal_{index}"},
        )
        for index in range(3)
    )
    threats = (
        Threat(
            entity_id=f"{episode_id}_entity_primary",
            priority=0.95,
            evidence_ids=(evidence[0].evidence_id,),
            patterns=("BACKLINE_ACCESS",),
        ),
        Threat(
            entity_id=f"{episode_id}_entity_secondary",
            priority=0.2,
            evidence_ids=(evidence[1].evidence_id,),
            patterns=("POKE",),
        ),
    )
    shared: dict[str, Any] = {
        "evidence_ids": (evidence[0].evidence_id,),
        "feasibility": 0.95,
        "supports_functions": frozenset({"PEEL"}),
        "win_condition_tags": frozenset({"PROTECT"}),
        "urgency_alignment": 0.9,
        "opportunity_cost": 0.05,
        "execution_burden": 0.05,
        "effects": ("PEEL",),
    }
    candidates = (
        CandidateAction(
            action_id=f"{episode_id}_action_initial",
            action_type=action_types[0],
            countered_threat_ids=frozenset({threats[0].entity_id}),
            equivalence_key="episode_initial_action",
            resource_required=900
            if template.task == "ITEMIZATION_DECISION"
            else None,
            **shared,
        ),
        CandidateAction(
            action_id=f"{episode_id}_action_followup",
            action_type=action_types[1],
            countered_threat_ids=frozenset({threats[1].entity_id}),
            equivalence_key="episode_followup_action",
            resource_required=1100
            if template.task == "ITEMIZATION_DECISION"
            else None,
            **shared,
        ),
    )
    required = frozenset({"temporalState", "teamPlan"})
    completeness = (
        0.5 if template.sequence_type == "REFRESH_RECOVERY" else 0.95
    )
    available = (
        frozenset({"teamPlan"})
        if template.sequence_type == "REFRESH_RECOVERY"
        else required
    )
    scenario = CanonicalScenario(
        scenario_id=f"{episode_id}_step_0",
        family_id=template.template_id,
        split_group=template.split_group,
        source_type="TEMPORAL",
        seed=seed,
        task=template.task,
        context=ScenarioContext(
            observed_at_game_second=600,
            freshness_seconds=1,
            completeness=completeness,
            required_fields=required,
            available_fields=available,
            state_signature="episode_state_initial",
        ),
        evidence=evidence,
        candidates=candidates,
        threats=threats,
        team_plan=TeamPlan(
            primary_win_condition="PROTECT_PLAN",
            win_condition_tags=frozenset({"PROTECT"}),
            missing_functions=frozenset({"PEEL"}),
            covered_functions=frozenset({"FRONTLINE"}),
        ),
        episode_id=episode_id,
        episode_step=0,
        causal_signature=f"{template.sequence_type}:initial",
    )
    return scenario


def _recent_advice(
    scenario: CanonicalScenario,
    result: OracleResult,
    age_seconds: int,
) -> tuple[RecentAdvice, ...]:
    selected_id = result.decision.primary_action_id
    if selected_id is None:
        return ()
    selected = next(
        candidate
        for candidate in scenario.candidates
        if candidate.action_id == selected_id
    )
    return (
        RecentAdvice(
            action_id=selected.action_id,
            equivalence_key=selected.equivalence_key,
            age_seconds=age_seconds,
            decision="SHOW",
            state_signature=scenario.context.state_signature,
            category=scenario.task,
            reason_codes=result.decision.reason_codes,
        ),
    )


def _advance_context(
    scenario: CanonicalScenario,
    elapsed_seconds: int,
    *,
    state_signature: str | None = None,
    completeness: float | None = None,
    available_fields: frozenset[str] | None = None,
) -> ScenarioContext:
    return replace(
        scenario.context,
        observed_at_game_second=600 + elapsed_seconds,
        freshness_seconds=1,
        state_signature=state_signature
        if state_signature is not None
        else scenario.context.state_signature,
        completeness=completeness
        if completeness is not None
        else scenario.context.completeness,
        available_fields=available_fields
        if available_fields is not None
        else scenario.context.available_fields,
    )


def _conflicting(
    evidence: Sequence[Evidence],
) -> tuple[Evidence, ...]:
    ids = [item.evidence_id for item in evidence[:3]]
    conflicts = {
        ids[0]: frozenset({ids[1]}),
        ids[1]: frozenset({ids[0], ids[2]}),
        ids[2]: frozenset({ids[1]}),
    }
    return tuple(
        replace(
            item,
            conflicts_with_evidence_ids=conflicts[item.evidence_id],
        )
        for item in evidence[:3]
    )


def _scenario_at(
    previous: CanonicalScenario,
    *,
    episode_id: str,
    step_index: int,
    elapsed_seconds: int,
    context: ScenarioContext,
    evidence: tuple[Evidence, ...] | None = None,
    candidates: tuple[CandidateAction, ...] | None = None,
    threats: tuple[Threat, ...] | None = None,
    recent_advice: tuple[RecentAdvice, ...] | None = None,
    signature: str,
) -> CanonicalScenario:
    return replace(
        previous,
        scenario_id=f"{episode_id}_step_{step_index}",
        parent_scenario_id=previous.scenario_id,
        episode_step=step_index,
        context=context,
        evidence=evidence if evidence is not None else previous.evidence,
        candidates=candidates
        if candidates is not None
        else previous.candidates,
        threats=threats if threats is not None else previous.threats,
        recent_advice=recent_advice
        if recent_advice is not None
        else previous.recent_advice,
        causal_signature=signature,
    )


def _apply_episode_variant(
    scenarios: tuple[CanonicalScenario, ...],
    seed: int,
) -> tuple[CanonicalScenario, ...]:
    variant = seed % 625
    pattern_band = variant % 5
    variant //= 5
    secondary_pattern_band = variant % 5
    variant //= 5
    primary_profile = variant % 5
    secondary_profile = variant // 5
    threat_patterns = (
        ("BACKLINE_ACCESS",),
        ("BURST",),
        ("HARD_CC",),
        ("LONG_RANGE",),
        ("RESET_PATTERN",),
    )
    damage_profiles = (
        (0.8, 0.15, 0.05),
        (0.15, 0.8, 0.05),
        (0.45, 0.45, 0.1),
        (0.3, 0.25, 0.45),
        (0.55, 0.25, 0.2),
    )
    primary_id = scenarios[0].threats[0].entity_id
    return tuple(
        replace(
            scenario,
            threats=tuple(
                replace(
                    threat,
                    damage_profile=(
                        damage_profiles[primary_profile]
                        if threat.entity_id == primary_id
                        else damage_profiles[secondary_profile]
                    ),
                    patterns=(
                        threat_patterns[pattern_band]
                        if threat.entity_id == primary_id
                        else threat_patterns[secondary_pattern_band]
                    ),
                )
                for threat in scenario.threats
            ),
        )
        for scenario in scenarios
    )


class EpisodeGenerator:
    def __init__(
        self,
        templates: Mapping[str, EpisodeTemplate] | None = None,
        oracle: StrategicOracle | None = None,
    ) -> None:
        self._templates = templates or load_episode_templates()
        self._oracle = oracle or StrategicOracle()

    @property
    def templates(self) -> Mapping[str, EpisodeTemplate]:
        return self._templates

    def generate(self, template_id: str, seed: int) -> Episode:
        try:
            template = self._templates[template_id]
        except KeyError as error:
            raise EpisodeError(
                f"template episodio sconosciuto: {template_id}"
            ) from error
        episode_id = _episode_id(template_id, seed)
        scenarios, deltas = self._materialize(template, episode_id, seed)
        steps: list[EpisodeStep] = []
        previous_result: OracleResult | None = None
        for index, scenario in enumerate(scenarios):
            result = self._oracle.decide(scenario)
            expected = template.expected_decisions[index]
            if result.decision.decision != expected:
                raise EpisodeError(
                    f"{template_id}[{index}]: decisione "
                    f"{result.decision.decision}, attesa {expected}"
                )
            transition = (
                f"START_TO_{expected}"
                if previous_result is None
                else f"{previous_result.decision.decision}_TO_{expected}"
            )
            steps.append(
                EpisodeStep(
                    index=index,
                    elapsed_seconds=template.elapsed_seconds[index],
                    previous_state=scenarios[index - 1]
                    if index > 0
                    else None,
                    delta=MappingProxyType(deltas[index]),
                    scenario=scenario,
                    expected_transition=transition,
                    previous_advice=previous_result.decision
                    if previous_result is not None
                    else None,
                    oracle_result=result,
                )
            )
            previous_result = result
        return Episode(
            episode_id=episode_id,
            template_id=template_id,
            split_name=_split_name(episode_id),
            split_group=template.split_group,
            steps=tuple(steps),
        )

    def _materialize(
        self,
        template: EpisodeTemplate,
        episode_id: str,
        seed: int,
    ) -> tuple[tuple[CanonicalScenario, ...], tuple[dict[str, object], ...]]:
        base = _base_scenario(template, episode_id, seed)
        first_result = self._oracle.decide(base)
        elapsed_1, elapsed_2 = template.elapsed_seconds[1:3]
        if template.sequence_type == "MACRO_CONTRADICTION":
            second = _scenario_at(
                base,
                episode_id=episode_id,
                step_index=1,
                elapsed_seconds=elapsed_1,
                context=_advance_context(base, elapsed_1),
                evidence=_conflicting(base.evidence),
                recent_advice=_recent_advice(base, first_result, elapsed_1),
                signature="macro:evidence_conflict",
            )
            third = _scenario_at(
                second,
                episode_id=episode_id,
                step_index=2,
                elapsed_seconds=elapsed_2,
                context=_advance_context(
                    second,
                    elapsed_2,
                    state_signature="episode_state_conflict_resolved",
                ),
                evidence=base.evidence,
                recent_advice=_recent_advice(base, first_result, elapsed_2),
                signature="macro:evidence_resolved",
            )
            fields_1 = ["context", "evidence", "recent_advice"]
            fields_2 = ["context", "evidence", "recent_advice"]
            events = ("CONTRADICTION_INTRODUCED", "CONTRADICTION_RESOLVED")
        elif template.sequence_type == "REFRESH_RECOVERY":
            second = _scenario_at(
                base,
                episode_id=episode_id,
                step_index=1,
                elapsed_seconds=elapsed_1,
                context=_advance_context(base, elapsed_1),
                signature="refresh:still_incomplete",
            )
            reliable = tuple(
                replace(
                    item,
                    confidence=0.98,
                    fact={"newReliableData": True, "signal": index},
                )
                for index, item in enumerate(base.evidence)
            )
            third = _scenario_at(
                second,
                episode_id=episode_id,
                step_index=2,
                elapsed_seconds=elapsed_2,
                context=_advance_context(
                    second,
                    elapsed_2,
                    state_signature="episode_state_complete",
                    completeness=0.95,
                    available_fields=second.context.required_fields,
                ),
                evidence=reliable,
                signature="refresh:reliable_data",
            )
            fields_1 = ["context"]
            fields_2 = ["context", "evidence"]
            events = ("CONTEXT_STILL_INCOMPLETE", "RELIABLE_DATA_ADDED")
        else:
            recent_1 = _recent_advice(base, first_result, elapsed_1)
            second = _scenario_at(
                base,
                episode_id=episode_id,
                step_index=1,
                elapsed_seconds=elapsed_1,
                context=_advance_context(base, elapsed_1),
                recent_advice=recent_1,
                signature=f"{template.sequence_type}:unchanged",
            )
            if template.sequence_type == "VALIDITY_EXPIRY":
                third = _scenario_at(
                    second,
                    episode_id=episode_id,
                    step_index=2,
                    elapsed_seconds=elapsed_2,
                    context=_advance_context(second, elapsed_2),
                    recent_advice=_recent_advice(
                        base, first_result, elapsed_2
                    ),
                    signature="validity:expired",
                )
                fields_2 = ["context", "recent_advice"]
                events = ("NO_SEMANTIC_CHANGE", "VALIDITY_PERIOD_EXPIRED")
            elif template.sequence_type == "ITEMIZATION_THREAT_CHANGE":
                first, second_threat = base.threats[:2]
                swapped = tuple(
                    replace(item, priority=second_threat.priority)
                    if item.entity_id == first.entity_id
                    else replace(item, priority=first.priority)
                    if item.entity_id == second_threat.entity_id
                    else item
                    for item in base.threats
                )
                third = _scenario_at(
                    second,
                    episode_id=episode_id,
                    step_index=2,
                    elapsed_seconds=elapsed_2,
                    context=_advance_context(
                        second,
                        elapsed_2,
                        state_signature="episode_state_threat_changed",
                    ),
                    threats=swapped,
                    recent_advice=_recent_advice(
                        base, first_result, elapsed_2
                    ),
                    signature="itemization:primary_threat_changed",
                )
                fields_2 = ["context", "threats", "recent_advice"]
                events = ("INVENTORY_UNCHANGED", "PRIMARY_THREAT_CHANGED")
            else:
                initial, followup = base.candidates[:2]
                changed_candidates = (
                    replace(initial, feasibility=0.0),
                    replace(
                        followup,
                        feasibility=1.0,
                        countered_threat_ids=initial.countered_threat_ids,
                    ),
                )
                state = (
                    "episode_state_vision_completed"
                    if template.sequence_type == "VISION_CONTEST"
                    else "episode_state_wave_changed"
                )
                third = _scenario_at(
                    second,
                    episode_id=episode_id,
                    step_index=2,
                    elapsed_seconds=elapsed_2,
                    context=_advance_context(
                        second, elapsed_2, state_signature=state
                    ),
                    candidates=changed_candidates,
                    recent_advice=_recent_advice(
                        base, first_result, elapsed_2
                    ),
                    signature=f"{template.sequence_type}:followup",
                )
                fields_2 = ["context", "candidates", "recent_advice"]
                events = (
                    ("NO_SEMANTIC_CHANGE", "VISION_COMPLETED")
                    if template.sequence_type == "VISION_CONTEST"
                    else ("WAVE_UNCHANGED", "WAVE_STATE_CHANGED")
                )
            fields_1 = ["context", "recent_advice"]
        deltas = (
            {
                "event": "INITIAL_STATE",
                "changedFields": [],
                "details": {"template": template.template_id},
            },
            {
                "event": events[0],
                "changedFields": fields_1,
                "details": {"elapsedSeconds": elapsed_1},
            },
            {
                "event": events[1],
                "changedFields": fields_2,
                "details": {"elapsedSeconds": elapsed_2},
            },
        )
        scenarios = _apply_episode_variant(
            (base, second, third), seed
        )
        return scenarios, deltas


def generate_episode(template_id: str, seed: int) -> Episode:
    return EpisodeGenerator().generate(template_id, seed)
