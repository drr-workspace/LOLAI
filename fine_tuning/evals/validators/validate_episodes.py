from __future__ import annotations

import argparse
from pathlib import Path

from evals.validators.validate_canonical import load_jsonl


def validate(
    scenarios_path: Path,
    episodes_path: Path,
    split_locations: dict[str, str] | None = None,
) -> list[str]:
    scenarios, errors = load_jsonl(scenarios_path)
    episodes, episode_errors = load_jsonl(episodes_path)
    errors.extend(episode_errors)
    by_id = {item.get("scenarioId"): item for item in scenarios}
    for episode_index, episode in enumerate(episodes, start=1):
        location = f"episodes.jsonl:{episode_index}"
        steps = episode.get("steps")
        if not isinstance(steps, list) or not 3 <= len(steps) <= 8:
            errors.append(f"{location}: servono da 3 a 8 step")
            continue
        elapsed = -1
        split: str | None = None
        previous: dict[str, object] | None = None
        for index, step in enumerate(steps):
            step_location = f"{location}.steps[{index}]"
            if not isinstance(step, dict):
                errors.append(f"{step_location}: oggetto atteso")
                continue
            scenario_id = step.get("scenarioId")
            scenario = by_id.get(scenario_id)
            if scenario is None:
                errors.append(f"{step_location}: scenario inesistente")
                continue
            if scenario.get("episodeId") != episode.get("episodeId"):
                errors.append(f"{step_location}: episodeId incoerente")
            if scenario.get("episodeStep") != index:
                errors.append(f"{step_location}: episodeStep non continuo")
            current_elapsed = step.get("elapsedSeconds")
            if not isinstance(current_elapsed, int) or current_elapsed <= elapsed:
                errors.append(f"{step_location}: tempo non monotono")
            else:
                elapsed = current_elapsed
            delta = step.get("delta")
            if not isinstance(delta, dict) or not delta:
                errors.append(f"{step_location}: delta mancante")
            transition = step.get("expectedTransition")
            decision = scenario.get("expectedOutput", {}).get("decision")
            if not isinstance(transition, str) or not transition.endswith(
                f"_TO_{decision}"
            ):
                errors.append(f"{step_location}: transizione incoerente")
            if previous is not None:
                advice = scenario.get("input", {}).get("recentAdvice")
                if not isinstance(advice, list):
                    errors.append(f"{step_location}: recentAdvice non valido")
            previous = scenario
            if split_locations is not None:
                current_split = split_locations.get(str(scenario_id))
                split = split or current_split
                if split != current_split:
                    errors.append(f"{location}: episodio spezzato fra split")
    return errors


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canonical-dir",
        type=Path,
        default=root / "datasets/canonical/releases/2.0.0",
    )
    args = parser.parse_args(argv)
    errors = validate(
        args.canonical_dir / "scenarios.jsonl",
        args.canonical_dir / "episodes.jsonl",
    )
    for error in errors:
        print(f"ERROR: {error}")
    print(f"Episodes: {'FAIL' if errors else 'PASS'}")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
