from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from typing import TypeVar


T = TypeVar("T")


class RandomSource:
    """Isolated deterministic random source; never touches module RNG state."""

    def __init__(self, seed: int) -> None:
        self._seed = seed
        self._random = random.Random(seed)

    @property
    def seed(self) -> int:
        return self._seed

    def for_scenario(self, scenario_seed: int) -> "RandomSource":
        material = f"{self._seed}:{scenario_seed}".encode("utf-8")
        derived = int.from_bytes(
            hashlib.sha256(material).digest()[:8], "big"
        )
        return RandomSource(derived)

    def randint(self, minimum: int, maximum: int) -> int:
        return self._random.randint(minimum, maximum)

    def uniform(self, minimum: float, maximum: float) -> float:
        return self._random.uniform(minimum, maximum)

    def choice(self, values: Sequence[T]) -> T:
        if not values:
            raise ValueError("choice richiede almeno un valore")
        return self._random.choice(values)

    def shuffle(self, values: Sequence[T]) -> tuple[T, ...]:
        shuffled = list(values)
        self._random.shuffle(shuffled)
        return tuple(shuffled)

    def sample(self, values: Sequence[T], count: int) -> tuple[T, ...]:
        return tuple(self._random.sample(values, count))

    def chance(self, probability: float) -> bool:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability deve essere compresa tra 0 e 1")
        return self._random.random() < probability
