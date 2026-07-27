from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ScenarioContext:
    observed_at_game_second: int
    freshness_seconds: int
    completeness: float
    uncertain_fields: tuple[str, ...] = ()
    required_fields: frozenset[str] = frozenset()
    available_fields: frozenset[str] = frozenset()
    state_signature: str = ""

    @property
    def missing_required_fields(self) -> frozenset[str]:
        return self.required_fields - self.available_fields


@dataclass(frozen=True, slots=True)
class Evidence:
    evidence_id: str
    category: str
    confidence: float
    freshness_seconds: int
    supports_action_ids: frozenset[str] = frozenset()
    conflicts_with_evidence_ids: frozenset[str] = frozenset()
    fact: object = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CandidateAction:
    action_id: str
    action_type: str
    evidence_ids: tuple[str, ...]
    feasibility: float
    supports_functions: frozenset[str] = frozenset()
    countered_threat_ids: frozenset[str] = frozenset()
    win_condition_tags: frozenset[str] = frozenset()
    urgency_alignment: float = 0.0
    opportunity_cost: float = 0.0
    execution_burden: float = 0.0
    equivalence_key: str = ""
    effects: tuple[str, ...] = ()
    resource_required: int | None = None


@dataclass(frozen=True, slots=True)
class Threat:
    entity_id: str
    priority: float
    evidence_ids: tuple[str, ...] = ()
    patterns: tuple[str, ...] = ()
    damage_profile: tuple[float, float, float] = (0.4, 0.5, 0.1)


@dataclass(frozen=True, slots=True)
class TeamPlan:
    primary_win_condition: str
    win_condition_tags: frozenset[str] = frozenset()
    missing_functions: frozenset[str] = frozenset()
    covered_functions: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class RecentAdvice:
    action_id: str
    equivalence_key: str
    age_seconds: int
    decision: str = "SHOW"
    state_signature: str = ""
    category: str = ""
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CanonicalScenario:
    scenario_id: str
    family_id: str
    split_group: str
    source_type: str
    seed: int
    task: str
    context: ScenarioContext
    evidence: tuple[Evidence, ...]
    candidates: tuple[CandidateAction, ...]
    threats: tuple[Threat, ...]
    team_plan: TeamPlan
    recent_advice: tuple[RecentAdvice, ...] = ()
    parent_scenario_id: str | None = None
    counterfactual_pair_id: str | None = None
    episode_id: str | None = None
    episode_step: int | None = None
    output_locale: str = "en-US"
    causal_signature: str = ""


@dataclass(frozen=True, slots=True)
class ScoreContribution:
    component_id: str
    raw_value: float
    direction: str
    weight: float
    contribution: float


@dataclass(frozen=True, slots=True)
class CandidateScore:
    action_id: str
    contributions: tuple[ScoreContribution, ...]
    unclamped_total: float
    total: float
    valid: bool


@dataclass(frozen=True, slots=True)
class ConfidenceContribution:
    factor_id: str
    raw_value: float
    direction: str
    weight: float
    contribution: float


@dataclass(frozen=True, slots=True)
class OracleDecision:
    schema_version: str
    decision: str
    category: str
    primary_action_id: str | None
    alternative_action_ids: tuple[str, ...]
    priority: str
    confidence: float
    reason_codes: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    valid_for_seconds: int
    recheck_triggers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OracleTrace:
    scenario_id: str
    seed: int
    gates_evaluated: tuple[str, ...]
    triggered_gate: str
    contradiction_count: int
    missing_required_fields: tuple[str, ...]
    candidate_scores: tuple[CandidateScore, ...]
    ranked_action_ids: tuple[str, ...]
    score_margin: float
    repeated_equivalence_key: str | None
    confidence_contributions: tuple[ConfidenceContribution, ...]
    confidence_unrounded: float
    final_decision: str
    selected_action_id: str | None
    reason_codes: tuple[str, ...]
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OracleResult:
    decision: OracleDecision
    trace: OracleTrace
