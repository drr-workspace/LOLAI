from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import replace
from types import MappingProxyType

from generators.domain_models import (
    CandidateAction,
    CanonicalScenario,
    Evidence,
    RecentAdvice,
    Threat,
)


ID_STYLES = frozenset({"neutral", "compact", "misleading"})


class IdFactory:
    """Produces deterministic opaque identifiers with no domain inference."""

    def __init__(self, seed: int, style: str = "neutral") -> None:
        if style not in ID_STYLES:
            raise ValueError(f"stile ID non supportato: {style}")
        self._seed = seed
        self._style = style

    def _token(self, namespace: str, key: str) -> str:
        digest = hashlib.blake2s(
            f"{self._seed}:{namespace}:{key}".encode("utf-8"),
            digest_size=7,
        ).hexdigest()
        if self._style == "compact":
            return digest
        if self._style == "misleading":
            decoys = ("priority", "safe", "stale", "optimal", "secondary")
            index = int(digest[:2], 16) % len(decoys)
            return f"{decoys[index]}_{digest}"
        return f"{namespace}_{digest}"

    def scenario_id(self, family_id: str, ordinal: int) -> str:
        return self._token("scenario", f"{family_id}:{ordinal}")

    def request_id(self, scenario_id: str) -> str:
        return self._token("request", scenario_id)

    def action_id(self, key: str) -> str:
        return self._token("action", key)

    def evidence_id(self, key: str) -> str:
        return self._token("evidence", key)

    def entity_id(self, key: str) -> str:
        return self._token("entity", key)

    def rename_scenario(self, scenario: CanonicalScenario) -> CanonicalScenario:
        action_map = MappingProxyType(
            {
                item.action_id: self.action_id(str(index))
                for index, item in enumerate(scenario.candidates)
            }
        )
        evidence_map = MappingProxyType(
            {
                item.evidence_id: self.evidence_id(str(index))
                for index, item in enumerate(scenario.evidence)
            }
        )
        entity_map = MappingProxyType(
            {
                item.entity_id: self.entity_id(str(index))
                for index, item in enumerate(scenario.threats)
            }
        )
        return replace(
            scenario,
            evidence=tuple(
                replace(
                    item,
                    evidence_id=evidence_map[item.evidence_id],
                    supports_action_ids=frozenset(
                        action_map.get(value, value)
                        for value in item.supports_action_ids
                    ),
                    conflicts_with_evidence_ids=frozenset(
                        evidence_map.get(value, value)
                        for value in item.conflicts_with_evidence_ids
                    ),
                )
                for item in scenario.evidence
            ),
            threats=tuple(
                replace(
                    item,
                    entity_id=entity_map[item.entity_id],
                    evidence_ids=_rename(item.evidence_ids, evidence_map),
                )
                for item in scenario.threats
            ),
            candidates=tuple(
                replace(
                    item,
                    action_id=action_map[item.action_id],
                    evidence_ids=_rename(item.evidence_ids, evidence_map),
                    countered_threat_ids=frozenset(
                        entity_map.get(value, value)
                        for value in item.countered_threat_ids
                    ),
                )
                for item in scenario.candidates
            ),
            recent_advice=tuple(
                replace(
                    item,
                    action_id=action_map.get(
                        item.action_id, self.action_id(f"recent:{index}")
                    ),
                )
                for index, item in enumerate(scenario.recent_advice)
            ),
        )


def _rename(
    values: Iterable[str], mapping: Mapping[str, str]
) -> tuple[str, ...]:
    return tuple(mapping.get(value, value) for value in values)
