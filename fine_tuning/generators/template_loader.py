from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias, cast

from jsonschema import Draft202012Validator

from generators.ontology_registry import (
    OntologyRegistry,
    default_ontology_dir,
)


FrozenJson: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | tuple["FrozenJson", ...]
    | Mapping[str, "FrozenJson"]
)

TEMPLATE_FILES: tuple[str, ...] = (
    "itemization-decision.json",
    "threat-assessment.json",
    "composition-plan.json",
    "matchup-plan.json",
    "macro-priority.json",
    "advice-suppression.json",
)


class TemplateError(ValueError):
    """Raised when a template document or family is invalid."""


class DuplicateFamilyIdError(TemplateError):
    """Raised when familyId is not globally unique."""


@dataclass(frozen=True, slots=True)
class TemplateFamily:
    family_id: str
    task: str
    description: str
    invariant_principle: str
    split_group: str
    source_eligibility: tuple[str, ...]
    parameter_ranges: Mapping[str, FrozenJson]
    required_context_fields: tuple[str, ...]
    evidence_blueprints: Mapping[str, FrozenJson]
    threat_blueprints: Mapping[str, FrozenJson]
    candidate_blueprints: Mapping[str, FrozenJson]
    recent_advice_blueprint: Mapping[str, FrozenJson]
    expected_decision_constraints: Mapping[str, FrozenJson]
    allowed_reason_codes: tuple[str, ...]
    counterfactual_axes: tuple[Mapping[str, FrozenJson], ...]
    adversarial_axes: tuple[Mapping[str, FrozenJson], ...]
    causal_parameters: tuple[Mapping[str, FrozenJson], ...]
    edge_case: Mapping[str, FrozenJson]
    useful_change: str
    insufficient_change: str
    semantic_comparison_fields: tuple[str, ...]
    reenabling_recheck_triggers: tuple[str, ...]
    message_intents: tuple[str, ...]
    minimum_examples: int
    maximum_examples: int
    template_file: str


def _freeze(value: Any) -> FrozenJson:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TemplateError(f"tipo JSON non supportato: {type(value).__name__}")


def _load_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TemplateError(f"{path}: deve essere un oggetto JSON")
    return value


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise TemplateError(f"{location}: deve essere una stringa non vuota")
    return value


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TemplateError(f"{location}: deve essere un oggetto")
    return value


def _array(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise TemplateError(f"{location}: deve essere una lista")
    return value


def _strings(value: Any, location: str) -> tuple[str, ...]:
    result = tuple(
        _string(item, f"{location}[{index}]")
        for index, item in enumerate(_array(value, location))
    )
    if len(set(result)) != len(result):
        raise TemplateError(f"{location}: contiene duplicati")
    return result


class TemplateRepository:
    """Validated, read-only access to all causal template families."""

    def __init__(
        self,
        template_dir: Path,
        ontology_dir: Path | None = None,
    ) -> None:
        self._template_dir = template_dir.resolve()
        self._ontology = OntologyRegistry(ontology_dir or default_ontology_dir())
        schema_path = self._template_dir / "template.schema.json"
        schema = _load_object(schema_path)
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)

        families: list[TemplateFamily] = []
        families_by_id: dict[str, TemplateFamily] = {}
        catalogs: dict[str, Mapping[str, FrozenJson]] = {}
        template_version: str | None = None

        for filename in TEMPLATE_FILES:
            path = self._template_dir / filename
            document = _load_object(path)
            errors = sorted(
                validator.iter_errors(document),
                key=lambda error: tuple(str(part) for part in error.absolute_path),
            )
            if errors:
                first = errors[0]
                json_path = ".".join(str(part) for part in first.absolute_path)
                raise TemplateError(f"{path}:{json_path}: {first.message}")

            current_version = _string(
                document.get("templateVersion"),
                f"{path}.templateVersion",
            )
            if template_version is None:
                template_version = current_version
            elif template_version != current_version:
                raise TemplateError(
                    f"{path}: templateVersion incoerente {current_version!r}"
                )
            if document.get("ontologyVersion") != self._ontology.version:
                raise TemplateError(f"{path}: ontologyVersion incoerente")

            document_task = _string(document.get("task"), f"{path}.task")
            if not self._ontology.contains("tasks", document_task):
                raise TemplateError(f"{path}: task non valido {document_task!r}")
            catalog = _object(
                document.get("blueprintCatalog"),
                f"{path}.blueprintCatalog",
            )
            catalogs[filename] = cast(
                Mapping[str, FrozenJson], _freeze(catalog)
            )

            for index, raw_family in enumerate(document["families"]):
                location = f"{path}.families[{index}]"
                family = self._parse_family(
                    _object(raw_family, location),
                    filename,
                    document_task,
                    catalog,
                    location,
                )
                if family.family_id in families_by_id:
                    raise DuplicateFamilyIdError(
                        f"{location}: familyId duplicato {family.family_id!r}"
                    )
                families_by_id[family.family_id] = family
                families.append(family)

        self._template_version = template_version or ""
        self._families = tuple(families)
        self._families_by_id = MappingProxyType(families_by_id)
        self._catalogs = MappingProxyType(catalogs)

    @property
    def template_version(self) -> str:
        return self._template_version

    @property
    def ontology_version(self) -> str:
        return self._ontology.version

    def require(self, family_id: str) -> TemplateFamily:
        try:
            return self._families_by_id[family_id]
        except KeyError as error:
            raise TemplateError(f"familyId sconosciuto: {family_id}") from error

    def families(
        self,
        *,
        task: str | None = None,
        split_group: str | None = None,
        source_eligibility: str | None = None,
    ) -> tuple[TemplateFamily, ...]:
        result = self._families
        if task is not None:
            result = tuple(family for family in result if family.task == task)
        if split_group is not None:
            result = tuple(
                family for family in result if family.split_group == split_group
            )
        if source_eligibility is not None:
            result = tuple(
                family
                for family in result
                if source_eligibility in family.source_eligibility
            )
        return result

    def resolve_profile(
        self,
        family: TemplateFamily,
        category: str,
        profile_name: str,
    ) -> FrozenJson:
        catalog = self._catalogs[family.template_file]
        category_profiles = catalog.get(category)
        if not isinstance(category_profiles, Mapping):
            raise TemplateError(
                f"{family.template_file}: categoria blueprint ignota {category}"
            )
        if profile_name not in category_profiles:
            raise TemplateError(
                f"{family.family_id}: profilo {category}/{profile_name} ignoto"
            )
        return category_profiles[profile_name]

    def _parse_family(
        self,
        raw: dict[str, Any],
        filename: str,
        document_task: str,
        catalog: Mapping[str, Any],
        location: str,
    ) -> TemplateFamily:
        family_id = _string(raw["familyId"], f"{location}.familyId")
        task = _string(raw["task"], f"{location}.task")
        if task != document_task:
            raise TemplateError(f"{location}: task diverso dal documento")

        source_eligibility = _strings(
            raw["sourceEligibility"], f"{location}.sourceEligibility"
        )
        for source_type in source_eligibility:
            if not self._ontology.contains("source-types", source_type):
                raise TemplateError(
                    f"{location}: sourceType inesistente {source_type!r}"
                )

        constraints = _object(
            raw["expectedDecisionConstraints"],
            f"{location}.expectedDecisionConstraints",
        )
        decisions = _strings(
            constraints["decisions"],
            f"{location}.expectedDecisionConstraints.decisions",
        )
        for decision in decisions:
            if not self._ontology.contains("decisions", decision):
                raise TemplateError(
                    f"{location}: decisione inesistente {decision!r}"
                )

        allowed_reasons = _strings(
            raw["allowedReasonCodes"], f"{location}.allowedReasonCodes"
        )
        self._validate_reason_compatibility(
            allowed_reasons, task, decisions, location
        )

        causal_parameters_raw = raw.get("causalParameters", [])
        causal_parameters = tuple(
            cast(Mapping[str, FrozenJson], _freeze(item))
            for item in _array(
                causal_parameters_raw, f"{location}.causalParameters"
            )
        )
        edge_case_raw = raw.get("edgeCase", {})
        edge_case = _object(edge_case_raw, f"{location}.edgeCase")
        detailed_tasks = {"COMPOSITION_PLAN", "MATCHUP_PLAN"}
        if task in detailed_tasks:
            if len(causal_parameters) < 3:
                raise TemplateError(
                    f"{location}: richiede almeno tre parametri causali"
                )
            if not edge_case:
                raise TemplateError(f"{location}: edgeCase obbligatorio")
            causal_ids = {
                str(parameter.get("id")) for parameter in causal_parameters
            }
            if edge_case.get("causalParameterId") not in causal_ids:
                raise TemplateError(
                    f"{location}: edgeCase non riferisce un parametro causale"
                )
            edge_constraints = _object(
                edge_case.get("expectedDecisionConstraints"),
                f"{location}.edgeCase.expectedDecisionConstraints",
            )
            edge_decisions = _strings(
                edge_constraints.get("decisions"),
                f"{location}.edgeCase.expectedDecisionConstraints.decisions",
            )
            edge_reasons = _strings(
                edge_case.get("allowedReasonCodes"),
                f"{location}.edgeCase.allowedReasonCodes",
            )
            self._validate_reason_compatibility(
                edge_reasons, task, edge_decisions, f"{location}.edgeCase"
            )

        semantic_fields = _strings(
            raw.get("semanticComparisonFields", []),
            f"{location}.semanticComparisonFields",
        )
        recheck_triggers = _strings(
            raw.get("reenablingRecheckTriggers", []),
            f"{location}.reenablingRecheckTriggers",
        )
        if task in {"MACRO_PRIORITY", "ADVICE_SUPPRESSION"}:
            _string(raw.get("usefulChange"), f"{location}.usefulChange")
            _string(
                raw.get("insufficientChange"),
                f"{location}.insufficientChange",
            )
            if not semantic_fields:
                raise TemplateError(
                    f"{location}: semanticComparisonFields obbligatorio"
                )
            if not recheck_triggers:
                raise TemplateError(
                    f"{location}: reenablingRecheckTriggers obbligatorio"
                )
            for trigger in recheck_triggers:
                if not self._ontology.contains("recheck-triggers", trigger):
                    raise TemplateError(
                        f"{location}: recheck trigger inesistente {trigger!r}"
                    )

        parameters = _object(
            raw["parameterRanges"], f"{location}.parameterRanges"
        )
        profile_references = {
            "contexts": parameters["contextProfile"],
            "teamPlans": parameters["teamPlanProfile"],
            "evidence": raw["evidenceBlueprints"]["profile"],
            "threats": raw["threatBlueprints"]["profile"],
            "candidates": raw["candidateBlueprints"]["profile"],
            "recentAdvice": raw["recentAdviceBlueprint"]["profile"],
        }
        for category, profile_name in profile_references.items():
            profiles = _object(catalog.get(category), f"{filename}.{category}")
            if profile_name not in profiles:
                raise TemplateError(
                    f"{location}: profilo inesistente {category}/{profile_name}"
                )
        self._validate_catalog_references(
            catalog,
            profile_references,
            location,
        )
        if edge_case:
            edge_profiles = {
                "contexts": edge_case["contextProfile"],
                "teamPlans": edge_case["teamPlanProfile"],
                "evidence": edge_case["evidenceProfile"],
                "threats": edge_case["threatProfile"],
                "candidates": edge_case["candidateProfile"],
                "recentAdvice": edge_case["recentAdviceProfile"],
            }
            for category, profile_name in edge_profiles.items():
                profiles = _object(
                    catalog.get(category), f"{filename}.{category}"
                )
                if profile_name not in profiles:
                    raise TemplateError(
                        f"{location}: profilo edge inesistente "
                        f"{category}/{profile_name}"
                    )
            self._validate_catalog_references(
                catalog, edge_profiles, f"{location}.edgeCase"
            )

        minimum_examples = raw["minimumExamples"]
        maximum_examples = raw["maximumExamples"]
        if minimum_examples > maximum_examples:
            raise TemplateError(
                f"{location}: minimumExamples supera maximumExamples"
            )

        return TemplateFamily(
            family_id=family_id,
            task=task,
            description=_string(raw["description"], f"{location}.description"),
            invariant_principle=_string(
                raw["invariantPrinciple"],
                f"{location}.invariantPrinciple",
            ),
            split_group=_string(raw["splitGroup"], f"{location}.splitGroup"),
            source_eligibility=source_eligibility,
            parameter_ranges=cast(
                Mapping[str, FrozenJson], _freeze(parameters)
            ),
            required_context_fields=_strings(
                raw["requiredContextFields"],
                f"{location}.requiredContextFields",
            ),
            evidence_blueprints=cast(
                Mapping[str, FrozenJson], _freeze(raw["evidenceBlueprints"])
            ),
            threat_blueprints=cast(
                Mapping[str, FrozenJson], _freeze(raw["threatBlueprints"])
            ),
            candidate_blueprints=cast(
                Mapping[str, FrozenJson], _freeze(raw["candidateBlueprints"])
            ),
            recent_advice_blueprint=cast(
                Mapping[str, FrozenJson],
                _freeze(raw["recentAdviceBlueprint"]),
            ),
            expected_decision_constraints=cast(
                Mapping[str, FrozenJson], _freeze(constraints)
            ),
            allowed_reason_codes=allowed_reasons,
            counterfactual_axes=cast(
                tuple[Mapping[str, FrozenJson], ...],
                _freeze(raw["counterfactualAxes"]),
            ),
            adversarial_axes=cast(
                tuple[Mapping[str, FrozenJson], ...],
                _freeze(raw["adversarialAxes"]),
            ),
            causal_parameters=causal_parameters,
            edge_case=cast(
                Mapping[str, FrozenJson], _freeze(edge_case)
            ),
            useful_change=str(raw.get("usefulChange", "")),
            insufficient_change=str(raw.get("insufficientChange", "")),
            semantic_comparison_fields=semantic_fields,
            reenabling_recheck_triggers=recheck_triggers,
            message_intents=_strings(
                raw["messageIntents"], f"{location}.messageIntents"
            ),
            minimum_examples=minimum_examples,
            maximum_examples=maximum_examples,
            template_file=filename,
        )

    def _validate_reason_compatibility(
        self,
        reasons: tuple[str, ...],
        task: str,
        decisions: tuple[str, ...],
        location: str,
    ) -> None:
        for reason in reasons:
            if not self._ontology.contains("reason-codes", reason):
                raise TemplateError(
                    f"{location}: reason code inesistente {reason!r}"
                )
            entry = self._ontology.require("reason-codes", reason)
            if task not in entry.get("allowedTasks", ()):
                raise TemplateError(
                    f"{location}: {reason} non compatibile con task {task}"
                )
            if not set(decisions) & set(
                cast(tuple[str, ...], entry.get("decisionCompatibility", ()))
            ):
                raise TemplateError(
                    f"{location}: {reason} incompatibile con le decisioni attese"
                )

    def _validate_catalog_references(
        self,
        catalog: Mapping[str, Any],
        profiles: Mapping[str, str],
        location: str,
    ) -> None:
        evidence = catalog["evidence"][profiles["evidence"]]
        evidence_roles = {
            item["role"] for item in evidence if isinstance(item, dict)
        }
        for item in evidence:
            if item["category"] not in self._ontology.ids("evidence-types"):
                raise TemplateError(
                    f"{location}: evidence type inesistente {item['category']!r}"
                )
            unknown_conflicts = set(item["conflictsWithRoles"]) - evidence_roles
            if unknown_conflicts:
                raise TemplateError(
                    f"{location}: conflict role inesistenti {unknown_conflicts}"
                )

        threats = catalog["threats"][profiles["threats"]]
        threat_roles = {item["role"] for item in threats}
        for item in threats:
            unknown_patterns = set(item["patterns"]) - self._ontology.ids(
                "threat-patterns"
            )
            if unknown_patterns:
                raise TemplateError(
                    f"{location}: threat pattern inesistenti {unknown_patterns}"
                )

        candidates = catalog["candidates"][profiles["candidates"]]
        candidate_roles = {item["role"] for item in candidates}
        for item in candidates:
            if item["actionType"] not in self._ontology.ids("action-types"):
                raise TemplateError(
                    f"{location}: action type inesistente {item['actionType']!r}"
                )
            unknown_evidence = set(item["evidenceRoles"]) - evidence_roles
            unknown_threats = set(item["counteredThreatRoles"]) - threat_roles
            unknown_functions = set(item["supportsFunctions"]) - self._ontology.ids(
                "team-functions"
            )
            if unknown_evidence or unknown_threats or unknown_functions:
                raise TemplateError(
                    f"{location}: riferimenti candidate non validi "
                    f"evidence={unknown_evidence}, threats={unknown_threats}, "
                    f"functions={unknown_functions}"
                )

        recent = catalog["recentAdvice"][profiles["recentAdvice"]]
        for item in recent:
            if item["actionRole"] not in candidate_roles:
                raise TemplateError(
                    f"{location}: recent action role inesistente "
                    f"{item['actionRole']!r}"
                )
            if item["decision"] not in self._ontology.ids("decisions"):
                raise TemplateError(
                    f"{location}: recent decision inesistente"
                )

        plan = catalog["teamPlans"][profiles["teamPlans"]]
        functions = set(plan["missingFunctions"]) | set(plan["coveredFunctions"])
        unknown_plan_functions = functions - self._ontology.ids("team-functions")
        if unknown_plan_functions:
            raise TemplateError(
                f"{location}: team functions inesistenti {unknown_plan_functions}"
            )


def default_template_dir() -> Path:
    return Path(__file__).resolve().parent / "templates"


def load_templates(
    template_dir: Path | None = None,
    ontology_dir: Path | None = None,
) -> TemplateRepository:
    return TemplateRepository(
        template_dir or default_template_dir(),
        ontology_dir=ontology_dir,
    )
