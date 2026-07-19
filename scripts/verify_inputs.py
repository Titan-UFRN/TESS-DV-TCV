#!/usr/bin/env python3
"""Verify the size-independent SHA-256 identity of paper input files."""

from __future__ import annotations

import argparse
import hashlib
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_checksums(path: pathlib.Path) -> list[tuple[str, pathlib.Path]]:
    entries: list[tuple[str, pathlib.Path]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"invalid checksum line {line_number}: {line!r}")
        entries.append((parts[0].lower(), pathlib.Path(parts[1].lstrip(" *"))))
    return entries


def verify(checksums: pathlib.Path, root: pathlib.Path) -> bool:
    ok = True
    for expected, relative in read_checksums(checksums):
        target = root / relative
        if not target.is_file():
            print(f"MISSING  {relative}")
            ok = False
            continue
        actual = sha256(target)
        if actual != expected:
            print(f"FAILED   {relative}")
            ok = False
            continue
        print(f"OK       {relative}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checksums", type=pathlib.Path, default=ROOT / "data/SHA256SUMS")
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    args = parser.parse_args()
    return 0 if verify(args.checksums, args.root) else 1


if __name__ == "__main__":
    raise SystemExit(main())
