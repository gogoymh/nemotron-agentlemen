"""Thin JSONL IO + path helpers. Deliberately keeps no runtime deps beyond stdlib + PyYAML."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml


def repo_root() -> Path:
    """`data-gen/` lives one level under the repo root."""
    return Path(__file__).resolve().parents[2]


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | os.PathLike[str]) -> Path:
    """Resolve a config-relative path against repo root unless already absolute."""
    p = Path(path)
    return p if p.is_absolute() else repo_root() / p


def iter_jsonl(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    path = resolve(path)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[dict[str, Any]]) -> int:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def count_jsonl(path: str | os.PathLike[str]) -> int:
    path = resolve(path)
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())
