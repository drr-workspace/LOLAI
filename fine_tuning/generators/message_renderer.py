from __future__ import annotations

import hashlib
from collections.abc import Mapping

from generators.domain_models import (
    CanonicalScenario,
    OracleDecision,
)


_FORMULATIONS: Mapping[str, tuple[str, ...]] = {
    "en-US": (
        "Prioritize {action}; it best fits the current evidence.",
        "Choose {action} now; it has the clearest strategic value.",
        "Use {action}; it is the strongest feasible option.",
        "Favor {action} while the current window remains valid.",
        "Commit to {action}; it best supports the declared plan.",
        "Take {action}; current evidence gives it the best margin.",
    ),
    "it-IT": (
        "Dai priorità a {action}: è l'opzione più coerente con le evidenze.",
        "Scegli {action} ora: offre il valore strategico più chiaro.",
        "Usa {action}: è l'alternativa fattibile più solida.",
        "Preferisci {action} finché questa finestra resta valida.",
        "Procedi con {action}: sostiene meglio il piano dichiarato.",
        "Esegui {action}: le evidenze attuali gli danno il margine migliore.",
    ),
}


class MessageRenderer:
    """Renders short controlled advice without an LLM."""

    maximum_length = 180

    def render(
        self,
        scenario: CanonicalScenario,
        decision: OracleDecision,
        *,
        intent: str = "",
    ) -> str:
        if decision.decision != "SHOW":
            return ""
        if decision.primary_action_id is None:
            raise ValueError("SHOW richiede primaryActionId")
        try:
            formulations = _FORMULATIONS[scenario.output_locale]
        except KeyError as error:
            raise ValueError(
                f"locale non supportato: {scenario.output_locale}"
            ) from error
        material = (
            f"{scenario.seed}:{scenario.family_id}:{intent}:"
            f"{decision.primary_action_id}"
        ).encode("utf-8")
        index = int.from_bytes(
            hashlib.sha256(material).digest()[:4], "big"
        ) % len(formulations)
        message = formulations[index].format(
            action=decision.primary_action_id
        )
        if len(message) > self.maximum_length:
            raise ValueError("messaggio oltre il limite di 180 caratteri")
        return message


def render_message(
    scenario: CanonicalScenario,
    decision: OracleDecision,
    *,
    intent: str = "",
) -> str:
    return MessageRenderer().render(scenario, decision, intent=intent)
