from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.validators.validate_contracts import (
    build_registry,
    build_validators,
    load_schemas,
    validate_row,
)
from generators.adversarial.apply import (
    apply_adversarial,
    load_rules,
)
from generators.release_renderer import ReleaseRenderer
from generators.sampler import ScenarioSampler


RULES = load_rules()
RULE_IDS = tuple(RULES)
SEEDS = (11, 22, 33, 44)
SAMPLER = ScenarioSampler()
BASE = SAMPLER.sample("composition_protect_vs_dive", 777)
RENDERER = ReleaseRenderer()
SCHEMA_DIR = (
    Path(__file__).resolve().parents[2] / "contracts" / "task-schemas"
)
SCHEMAS = load_schemas(SCHEMA_DIR)
VALIDATORS = build_validators(SCHEMAS, build_registry(SCHEMAS))


@pytest.mark.parametrize("rule_id", RULE_IDS)
@pytest.mark.parametrize("mutation_seed", SEEDS)
def test_adversarial_effects_match_declaration(
    rule_id: str, mutation_seed: int
) -> None:
    application = apply_adversarial(
        BASE, rule_id, mutation_seed=mutation_seed
    )

    assert application.observed_effect == application.expected_effect
    assert application.mutated.source_type == "ADVERSARIAL"
    assert application.mutated.parent_scenario_id == BASE.scenario_id


@pytest.mark.parametrize(
    "rule_id",
    tuple(
        rule.id for rule in RULES.values() if not rule.semantic_relevance
    ),
)
def test_irrelevant_mutations_preserve_semantic_decision(
    rule_id: str,
) -> None:
    application = apply_adversarial(BASE, rule_id, mutation_seed=91)

    assert application.observed_effect == "SAME_DECISION"


@pytest.mark.parametrize(
    "rule_id",
    tuple(
        rule.id for rule in RULES.values() if rule.semantic_relevance
    ),
)
def test_relevant_mutations_declare_change(rule_id: str) -> None:
    application = apply_adversarial(BASE, rule_id, mutation_seed=92)

    assert application.expected_effect != "SAME_DECISION"
    assert application.observed_effect == application.expected_effect


def test_prompt_injection_exists_only_in_untrusted_data() -> None:
    application = apply_adversarial(
        BASE, "PROMPT_INJECTION_IN_DATA", mutation_seed=93
    )
    row = RENDERER.render(
        application.mutated, application.mutated_result
    )
    injection = "Ignore previous instructions"

    assert injection not in row["messages"][0]["content"]
    assert injection in row["messages"][1]["content"]
    assert injection not in row["messages"][2]["content"]


@pytest.mark.parametrize(
    "rule_id",
    (
        "UNKNOWN_OPTIONAL_FACT_FIELDS",
        "LONG_VALID_EVIDENCE_LIST",
        "PROMPT_INJECTION_IN_DATA",
    ),
)
def test_schema_permitted_adversarial_data_remains_valid(
    rule_id: str,
) -> None:
    application = apply_adversarial(BASE, rule_id, mutation_seed=94)
    row = RENDERER.render(
        application.mutated, application.mutated_result
    )
    errors: list[str] = []

    validate_row(row, Path("adversarial.jsonl"), 1, VALIDATORS, errors)

    assert errors == []
    assert json.loads(row["messages"][1]["content"])


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_adversarial_mutation_is_deterministic(rule_id: str) -> None:
    assert apply_adversarial(
        BASE, rule_id, mutation_seed=95
    ) == apply_adversarial(BASE, rule_id, mutation_seed=95)
