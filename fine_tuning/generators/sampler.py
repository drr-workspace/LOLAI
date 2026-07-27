from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

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
from generators.id_factory import IdFactory
from generators.oracle import StrategicOracle
from generators.policy_loader import PolicyBundle, load_policies
from generators.random_source import RandomSource
from generators.template_loader import (
    TemplateFamily,
    TemplateRepository,
    load_templates,
)


SUPPORTED_LOCALES = frozenset({"it-IT", "en-US"})


@dataclass(frozen=True, slots=True)
class SampledScenario:
    scenario: CanonicalScenario
    oracle_result: OracleResult
    message_intent: str


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location}: atteso oggetto")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, location: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{location}: attesa lista")
    return cast(Sequence[Any], value)


def _profile(reference: Mapping[str, object]) -> str:
    value = reference.get("profile")
    if not isinstance(value, str):
        raise ValueError("riferimento profilo non valido")
    return value


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _causal_choices(
    values: Sequence[object], *, field: str
) -> tuple[object, ...]:
    if (
        len(values) == 2
        and _is_number(values[0])
        and _is_number(values[1])
    ):
        lower = float(values[0])
        upper = float(values[1])
        if (
            field == "context.completeness"
            and lower < 0.7 <= upper
        ):
            return (
                round(lower, 3),
                round(max(0.72, lower + (upper - lower) * 0.4), 3),
                round(max(0.78, lower + (upper - lower) * 0.6), 3),
                round(max(0.86, lower + (upper - lower) * 0.8), 3),
                round(upper, 3),
            )
        return tuple(
            round(lower + (upper - lower) * index / 4, 3)
            for index in range(5)
        )
    return tuple(values)


class ScenarioSampler:
    """Materializes deterministic scenarios from declarative families."""

    def __init__(
        self,
        repository: TemplateRepository | None = None,
        policies: PolicyBundle | None = None,
        oracle: StrategicOracle | None = None,
    ) -> None:
        self._repository = repository or load_templates()
        self._policies = policies or load_policies()
        self._oracle = oracle or StrategicOracle(policies=self._policies)
        self._random = RandomSource(self._policies.generation.global_seed)

    def sample(
        self,
        family: TemplateFamily | str,
        scenario_seed: int,
        *,
        output_locale: str = "en-US",
        ordinal: int = 0,
        id_style: str = "neutral",
    ) -> CanonicalScenario:
        return self.sample_with_result(
            family,
            scenario_seed,
            output_locale=output_locale,
            ordinal=ordinal,
            id_style=id_style,
        ).scenario

    def sample_with_result(
        self,
        family: TemplateFamily | str,
        scenario_seed: int,
        *,
        output_locale: str = "en-US",
        ordinal: int = 0,
        id_style: str = "neutral",
    ) -> SampledScenario:
        if output_locale not in SUPPORTED_LOCALES:
            raise ValueError(f"locale non supportato: {output_locale}")
        selected = (
            self._repository.require(family)
            if isinstance(family, str)
            else family
        )
        rng = self._random.for_scenario(scenario_seed)
        ids = IdFactory(scenario_seed, style=id_style)
        evidence, evidence_by_role = self._build_evidence(selected, rng, ids)
        threats, threat_by_role = self._build_threats(
            selected, rng, ids, evidence
        )
        candidates, action_by_role = self._build_candidates(
            selected,
            rng,
            ids,
            evidence,
            evidence_by_role,
            threat_by_role,
        )
        context = self._build_context(selected, rng)
        recent = self._build_recent_advice(
            selected, ids, action_by_role, context
        )
        team_plan = self._build_team_plan(selected)
        scenario_id = ids.scenario_id(selected.family_id, ordinal)
        source_type = rng.choice(selected.source_eligibility)
        provisional = CanonicalScenario(
            scenario_id=scenario_id,
            family_id=selected.family_id,
            split_group=selected.split_group,
            source_type=source_type,
            seed=scenario_seed,
            task=selected.task,
            context=context,
            evidence=evidence,
            candidates=rng.shuffle(candidates),
            threats=rng.shuffle(threats),
            team_plan=team_plan,
            recent_advice=recent,
            output_locale=output_locale,
        )
        provisional = self._apply_causal_parameters(
            provisional,
            selected,
            ordinal,
            action_by_role=action_by_role,
            threat_by_role=threat_by_role,
        )
        signature = self._causal_signature(provisional)
        scenario = replace(provisional, causal_signature=signature)
        result = self._oracle.decide(scenario)
        return SampledScenario(
            scenario=scenario,
            oracle_result=result,
            message_intent=rng.choice(selected.message_intents),
        )

    def _apply_causal_parameters(
        self,
        scenario: CanonicalScenario,
        family: TemplateFamily,
        ordinal: int,
        *,
        action_by_role: Mapping[str, str],
        threat_by_role: Mapping[str, str],
    ) -> CanonicalScenario:
        """Materialize deterministic combinations declared by the family.

        Numeric endpoints are interpolated across the same five bands used by
        causal signatures. This creates semantic variation instead of merely
        changing opaque identifiers or collection order.
        """

        if ordinal == 0:
            return scenario
        result = self._apply_baseline_causal_grid(
            scenario,
            ordinal - 1,
            action_by_role=action_by_role,
            threat_by_role=threat_by_role,
        )
        variant = ordinal - 1
        for parameter in family.causal_parameters:
            field = parameter.get("field")
            values = parameter.get("values")
            if not isinstance(field, str) or not isinstance(values, tuple):
                continue
            choices = _causal_choices(values, field=field)
            if not choices:
                continue
            value = choices[variant % len(choices)]
            variant //= len(choices)
            result = self._apply_causal_value(
                result,
                field,
                value,
                action_by_role=action_by_role,
                threat_by_role=threat_by_role,
            )
        return result

    def _apply_baseline_causal_grid(
        self,
        scenario: CanonicalScenario,
        variant: int,
        *,
        action_by_role: Mapping[str, str],
        threat_by_role: Mapping[str, str],
    ) -> CanonicalScenario:
        feasibility_mode = variant % 5
        variant //= 5
        if feasibility_mode == 0:
            candidates = tuple(
                replace(candidate, feasibility=0.02)
                for candidate in scenario.candidates
            )
        else:
            preferred_id = action_by_role.get("preferred")
            preferred_value = (0.45, 0.62, 0.8, 0.95)[
                feasibility_mode - 1
            ]
            candidates = tuple(
                replace(candidate, feasibility=preferred_value)
                if candidate.action_id == preferred_id
                else candidate
                for candidate in scenario.candidates
            )
        result = replace(scenario, candidates=candidates)
        result = self._apply_causal_value(
            result,
            "threats.primary.priority",
            (0.35, 0.5, 0.65, 0.8, 0.95)[variant % 5],
            action_by_role=action_by_role,
            threat_by_role=threat_by_role,
        )
        variant //= 5
        result = self._apply_causal_value(
            result,
            "candidates.preferred.urgencyAlignment",
            (0.25, 0.45, 0.65, 0.82, 0.95)[variant % 5],
            action_by_role=action_by_role,
            threat_by_role=threat_by_role,
        )
        variant //= 5
        preferred_id = action_by_role.get("preferred")
        opportunity_cost = (0.05, 0.2, 0.4, 0.62, 0.85)[
            variant % 5
        ]
        return replace(
            result,
            candidates=tuple(
                replace(candidate, opportunity_cost=opportunity_cost)
                if candidate.action_id == preferred_id
                else candidate
                for candidate in result.candidates
            ),
        )

    def _apply_causal_value(
        self,
        scenario: CanonicalScenario,
        field: str,
        value: object,
        *,
        action_by_role: Mapping[str, str],
        threat_by_role: Mapping[str, str],
    ) -> CanonicalScenario:
        if field == "context.completeness" and _is_number(value):
            return replace(
                scenario,
                context=replace(
                    scenario.context,
                    completeness=round(float(value), 3),
                ),
            )
        candidate_fields = {
            "feasibility": "feasibility",
            "executionBurden": "execution_burden",
            "urgencyAlignment": "urgency_alignment",
        }
        for role in ("preferred", "alternative"):
            prefix = f"candidates.{role}."
            attribute = candidate_fields.get(field.removeprefix(prefix))
            if (
                field.startswith(prefix)
                and attribute is not None
                and _is_number(value)
            ):
                action_id = action_by_role.get(role)
                return replace(
                    scenario,
                    candidates=tuple(
                        replace(
                            candidate,
                            **{attribute: round(float(value), 3)},
                        )
                        if candidate.action_id == action_id
                        else candidate
                        for candidate in scenario.candidates
                    ),
                )
        if field == "threats.primary.priority" and _is_number(value):
            entity_id = threat_by_role.get("primary")
            return replace(
                scenario,
                threats=tuple(
                    replace(threat, priority=round(float(value), 3))
                    if threat.entity_id == entity_id
                    else threat
                    for threat in scenario.threats
                ),
            )
        return scenario

    def _base_profile(
        self,
        family: TemplateFamily,
        category: str,
        profile_name: str,
    ) -> object:
        return self._repository.resolve_profile(
            family, category, profile_name
        )

    def _build_context(
        self, family: TemplateFamily, rng: RandomSource
    ) -> ScenarioContext:
        profile_name = str(family.parameter_ranges["contextProfile"])
        raw = _mapping(
            self._base_profile(family, "contexts", profile_name),
            f"{family.family_id}.context",
        )
        completeness = round(
            max(0.0, min(1.0, float(raw["completeness"]) + rng.uniform(-0.02, 0.02))),
            3,
        )
        return ScenarioContext(
            observed_at_game_second=int(raw["observedAtGameSecond"]),
            freshness_seconds=int(raw["freshnessSeconds"]),
            completeness=completeness,
            uncertain_fields=tuple(raw["uncertainFields"]),
            required_fields=frozenset(family.required_context_fields),
            available_fields=frozenset(raw["availableFields"]),
            state_signature=str(raw["stateSignature"]),
        )

    def _build_evidence(
        self,
        family: TemplateFamily,
        rng: RandomSource,
        ids: IdFactory,
    ) -> tuple[tuple[Evidence, ...], Mapping[str, str]]:
        raw_items = _sequence(
            self._base_profile(
                family, "evidence", _profile(family.evidence_blueprints)
            ),
            f"{family.family_id}.evidence",
        )
        maximum = self._policies.generation.evidence_count.maximum
        target = rng.randint(
            max(len(raw_items), self._policies.generation.evidence_count.minimum),
            max(len(raw_items), min(maximum, len(raw_items) + 2)),
        )
        expanded = list(raw_items)
        while len(expanded) < target:
            expanded.append(raw_items[(len(expanded) - len(raw_items)) % len(raw_items)])
        role_ids = {
            str(_mapping(item, "evidence").get("role")): ids.evidence_id(
                f"role:{_mapping(item, 'evidence').get('role')}"
            )
            for item in raw_items
        }
        evidence: list[Evidence] = []
        for index, value in enumerate(expanded):
            item = _mapping(value, f"{family.family_id}.evidence")
            role = str(item["role"])
            evidence_id = (
                role_ids[role]
                if index < len(raw_items)
                else ids.evidence_id(f"extra:{index}")
            )
            evidence.append(
                Evidence(
                    evidence_id=evidence_id,
                    category=str(item["category"]),
                    confidence=float(item["confidence"]),
                    freshness_seconds=int(item["freshnessSeconds"]),
                    conflicts_with_evidence_ids=frozenset(
                        role_ids[conflict]
                        for conflict in item["conflictsWithRoles"]
                        if conflict in role_ids
                    ),
                    fact={
                        "role": f"signal_{index}",
                        "stateSignature": role,
                    },
                )
            )
        return tuple(evidence), role_ids

    def _build_threats(
        self,
        family: TemplateFamily,
        rng: RandomSource,
        ids: IdFactory,
        evidence: tuple[Evidence, ...],
    ) -> tuple[tuple[Threat, ...], Mapping[str, str]]:
        raw_items = _sequence(
            self._base_profile(
                family, "threats", _profile(family.threat_blueprints)
            ),
            f"{family.family_id}.threats",
        )
        target = rng.randint(len(raw_items), min(3, len(raw_items) + 1))
        expanded = list(raw_items)
        while len(expanded) < target:
            expanded.append(raw_items[0])
        role_ids = {
            str(_mapping(item, "threat").get("role")): ids.entity_id(
                f"role:{_mapping(item, 'threat').get('role')}"
            )
            for item in raw_items
        }
        threats: list[Threat] = []
        for index, value in enumerate(expanded):
            item = _mapping(value, f"{family.family_id}.threat")
            role = str(item["role"])
            entity_id = (
                role_ids[role]
                if index < len(raw_items)
                else ids.entity_id(f"extra:{index}")
            )
            priority = max(0.0, float(item["priority"]) - index * 0.12)
            threats.append(
                Threat(
                    entity_id=entity_id,
                    priority=round(priority, 3),
                    evidence_ids=(evidence[index % len(evidence)].evidence_id,),
                    patterns=tuple(item["patterns"]),
                )
            )
        return tuple(threats), role_ids

    def _build_candidates(
        self,
        family: TemplateFamily,
        rng: RandomSource,
        ids: IdFactory,
        evidence: tuple[Evidence, ...],
        evidence_by_role: Mapping[str, str],
        threat_by_role: Mapping[str, str],
    ) -> tuple[tuple[CandidateAction, ...], Mapping[str, str]]:
        raw_items = _sequence(
            self._base_profile(
                family, "candidates", _profile(family.candidate_blueprints)
            ),
            f"{family.family_id}.candidates",
        )
        bounds = self._policies.generation.candidate_count
        target = rng.randint(
            max(bounds.minimum, len(raw_items)),
            max(len(raw_items), min(bounds.maximum, len(raw_items) + 2)),
        )
        expanded = list(raw_items)
        while len(expanded) < target:
            expanded.append(raw_items[-1])
        role_ids = {
            str(_mapping(item, "candidate").get("role")): ids.action_id(
                f"role:{_mapping(item, 'candidate').get('role')}"
            )
            for item in raw_items
        }
        candidates: list[CandidateAction] = []
        for index, value in enumerate(expanded):
            item = _mapping(value, f"{family.family_id}.candidate")
            role = str(item["role"])
            action_id = (
                role_ids[role]
                if index < len(raw_items)
                else ids.action_id(f"extra:{index}")
            )
            evidence_ids = tuple(
                evidence_by_role[ref]
                for ref in item["evidenceRoles"]
                if ref in evidence_by_role
            ) or (evidence[index % len(evidence)].evidence_id,)
            feasibility = float(item["feasibility"])
            if index >= len(raw_items):
                feasibility = max(0.0, feasibility - 0.15 * index)
            candidates.append(
                CandidateAction(
                    action_id=action_id,
                    action_type=str(item["actionType"]),
                    evidence_ids=evidence_ids,
                    feasibility=round(feasibility, 3),
                    supports_functions=frozenset(item["supportsFunctions"]),
                    countered_threat_ids=frozenset(
                        threat_by_role[ref]
                        for ref in item["counteredThreatRoles"]
                        if ref in threat_by_role
                    ),
                    win_condition_tags=frozenset(item["winConditionTags"]),
                    urgency_alignment=float(item["urgencyAlignment"]),
                    opportunity_cost=float(item["opportunityCost"]),
                    execution_burden=float(item["executionBurden"]),
                    equivalence_key=str(item["equivalenceKey"]),
                    effects=tuple(sorted(item["supportsFunctions"])),
                    resource_required=(
                        rng.randint(0, 1800)
                        if family.task == "ITEMIZATION_DECISION"
                        else None
                    ),
                )
            )
        return tuple(candidates), role_ids

    def _build_team_plan(self, family: TemplateFamily) -> TeamPlan:
        profile_name = str(family.parameter_ranges["teamPlanProfile"])
        raw = _mapping(
            self._base_profile(family, "teamPlans", profile_name),
            f"{family.family_id}.teamPlan",
        )
        return TeamPlan(
            primary_win_condition=str(raw["primaryWinCondition"]),
            win_condition_tags=frozenset(raw["winConditionTags"]),
            missing_functions=frozenset(raw["missingFunctions"]),
            covered_functions=frozenset(raw["coveredFunctions"]),
        )

    def _build_recent_advice(
        self,
        family: TemplateFamily,
        ids: IdFactory,
        action_by_role: Mapping[str, str],
        context: ScenarioContext,
    ) -> tuple[RecentAdvice, ...]:
        raw_items = _sequence(
            self._base_profile(
                family,
                "recentAdvice",
                _profile(family.recent_advice_blueprint),
            ),
            f"{family.family_id}.recentAdvice",
        )
        return tuple(
            RecentAdvice(
                action_id=action_by_role.get(
                    str(item["actionRole"]),
                    ids.action_id(f"recent:{index}"),
                ),
                equivalence_key=str(item["equivalenceKey"]),
                age_seconds=int(item["ageSeconds"]),
                decision=str(item["decision"]),
                state_signature=str(item.get("stateSignature", context.state_signature)),
                category=family.task,
                reason_codes=("RECENTLY_ADVISED",),
            )
            for index, raw in enumerate(raw_items)
            for item in [_mapping(raw, f"{family.family_id}.recentAdvice")]
        )

    def _causal_signature(self, scenario: CanonicalScenario) -> str:
        payload = {
            "task": scenario.task,
            "completeness": scenario.context.completeness,
            "freshness": scenario.context.freshness_seconds,
            "candidateTypes": sorted(
                item.action_type for item in scenario.candidates
            ),
            "teamFunctions": sorted(scenario.team_plan.missing_functions),
            "threatPatterns": sorted(
                pattern
                for threat in scenario.threats
                for pattern in threat.patterns
            ),
        }
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def sample_scenario(
    family: TemplateFamily | str,
    scenario_seed: int,
    *,
    output_locale: str = "en-US",
    ordinal: int = 0,
    id_style: str = "neutral",
) -> CanonicalScenario:
    return ScenarioSampler().sample(
        family,
        scenario_seed,
        output_locale=output_locale,
        ordinal=ordinal,
        id_style=id_style,
    )
