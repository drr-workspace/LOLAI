from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias, cast


FrozenValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | tuple["FrozenValue", ...]
    | Mapping[str, "FrozenValue"]
)
OntologyEntry: TypeAlias = Mapping[str, FrozenValue]


class OntologyError(ValueError):
    """Base error raised for invalid ontology data."""


class DuplicateOntologyIdError(OntologyError):
    """Raised when a registry contains the same ID more than once."""


class InconsistentOntologyVersionError(OntologyError):
    """Raised when ontology files do not share one version."""


class UnknownOntologyRegistryError(OntologyError):
    """Raised when a requested registry is not loaded."""


class UnknownOntologyValueError(OntologyError):
    """Raised when a requested ID is absent from a registry."""


def _load_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise OntologyError(f"{path}: il documento deve essere un oggetto JSON")
    return document


def _required_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise OntologyError(f"{location}: deve essere una stringa non vuota")
    return value


def _required_values(document: Mapping[str, Any], path: Path) -> list[Any]:
    values = document.get("values")
    if not isinstance(values, list):
        raise OntologyError(f"{path}: values deve essere una lista")
    return values


def _deep_freeze(value: Any) -> FrozenValue:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _deep_freeze(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(child) for child in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise OntologyError(f"tipo JSON non supportato: {type(value).__name__}")


class OntologyRegistry:
    """Read-only access to all ontology registries declared by the manifest."""

    def __init__(self, ontology_dir: Path) -> None:
        self._ontology_dir = ontology_dir.resolve()
        manifest_path = self._ontology_dir / "manifest.json"
        manifest = _load_object(manifest_path)
        self._version = _required_string(
            manifest.get("ontologyVersion"),
            f"{manifest_path}: ontologyVersion",
        )

        registry_values = _required_values(manifest, manifest_path)
        manifest_ids: set[str] = set()
        loaded: dict[str, Mapping[str, OntologyEntry]] = {}
        ordered: dict[str, tuple[OntologyEntry, ...]] = {}

        for index, raw_entry in enumerate(registry_values):
            location = f"{manifest_path}: values[{index}]"
            if not isinstance(raw_entry, dict):
                raise OntologyError(f"{location}: deve essere un oggetto")
            registry_name = _required_string(raw_entry.get("id"), f"{location}.id")
            filename = _required_string(raw_entry.get("file"), f"{location}.file")
            if registry_name in manifest_ids:
                raise DuplicateOntologyIdError(
                    f"{manifest_path}: ID manifest duplicato: {registry_name}"
                )
            manifest_ids.add(registry_name)
            entries_by_id, entries = self._load_registry(
                registry_name,
                self._ontology_dir / filename,
            )
            loaded[registry_name] = MappingProxyType(entries_by_id)
            ordered[registry_name] = tuple(entries)

        self._registries = MappingProxyType(loaded)
        self._ordered_values = MappingProxyType(ordered)

    @property
    def version(self) -> str:
        """Return the common ontology version."""
        return self._version

    @property
    def registry_names(self) -> tuple[str, ...]:
        """Return loaded registry names in manifest order."""
        return tuple(self._registries)

    def contains(self, registry_name: str, value_id: str) -> bool:
        """Return whether an ID exists in the selected registry."""
        return value_id in self._registry(registry_name)

    def require(self, registry_name: str, value_id: str) -> OntologyEntry:
        """Return an immutable entry or fail when its ID does not exist."""
        registry = self._registry(registry_name)
        try:
            return registry[value_id]
        except KeyError as error:
            raise UnknownOntologyValueError(
                f"ID ontologico sconosciuto in {registry_name}: {value_id}"
            ) from error

    def values(self, registry_name: str) -> tuple[OntologyEntry, ...]:
        """Return immutable entries in source order."""
        self._registry(registry_name)
        return self._ordered_values[registry_name]

    def ids(self, registry_name: str) -> frozenset[str]:
        """Return the immutable set of IDs in a registry."""
        return frozenset(self._registry(registry_name))

    def _registry(self, registry_name: str) -> Mapping[str, OntologyEntry]:
        try:
            return self._registries[registry_name]
        except KeyError as error:
            raise UnknownOntologyRegistryError(
                f"registro ontologico sconosciuto: {registry_name}"
            ) from error

    def _load_registry(
        self,
        registry_name: str,
        path: Path,
    ) -> tuple[dict[str, OntologyEntry], list[OntologyEntry]]:
        document = _load_object(path)
        version = _required_string(
            document.get("ontologyVersion"),
            f"{path}: ontologyVersion",
        )
        if version != self._version:
            raise InconsistentOntologyVersionError(
                f"{path}: versione {version!r}, attesa {self._version!r}"
            )

        entries_by_id: dict[str, OntologyEntry] = {}
        ordered_entries: list[OntologyEntry] = []
        for index, raw_entry in enumerate(_required_values(document, path)):
            location = f"{path}: values[{index}]"
            if not isinstance(raw_entry, dict):
                raise OntologyError(f"{location}: deve essere un oggetto")
            value_id = _required_string(raw_entry.get("id"), f"{location}.id")
            if value_id in entries_by_id:
                raise DuplicateOntologyIdError(
                    f"{path}: ID duplicato nel registro {registry_name}: {value_id}"
                )
            frozen = cast(OntologyEntry, _deep_freeze(raw_entry))
            entries_by_id[value_id] = frozen
            ordered_entries.append(frozen)
        return entries_by_id, ordered_entries


def default_ontology_dir() -> Path:
    """Return the repository ontology directory."""
    return Path(__file__).resolve().parents[1] / "ontology"


def load_ontology(ontology_dir: Path | None = None) -> OntologyRegistry:
    """Load the complete ontology from an explicit or default directory."""
    return OntologyRegistry(ontology_dir or default_ontology_dir())
