from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_lines(
    paths: Iterable[Path], *, relative_to: Path
) -> tuple[str, ...]:
    return tuple(
        f"{sha256_file(path)}  {path.relative_to(relative_to).as_posix()}"
        for path in sorted(paths)
    )


def write_checksums(
    output_path: Path,
    paths: Iterable[Path],
    *,
    relative_to: Path | None = None,
) -> None:
    root = relative_to or output_path.parent
    lines = checksum_lines(paths, relative_to=root)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_checksums(checksum_path: Path) -> tuple[str, ...]:
    errors: list[str] = []
    root = checksum_path.parent
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            errors.append(f"riga {line_number}: formato non valido")
            continue
        path = root / relative
        if not path.is_file():
            errors.append(f"riga {line_number}: file mancante {relative}")
        elif sha256_file(path) != expected:
            errors.append(f"riga {line_number}: checksum errato {relative}")
    return tuple(errors)
