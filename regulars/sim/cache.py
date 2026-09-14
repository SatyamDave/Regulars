"""Sim result cache (§8: recompute only on persona version bump or new candidates).

Keyed on the two persona hashes, scenario, seed and model: any change to a spec changes its hash
and misses; identical inputs hit and skip the LLM entirely. In-memory for tests and single runs;
append-only JSONL on disk for anything that should survive a process.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from regulars.core.transcript import Transcript


@dataclass(frozen=True)
class SimCacheKey:
    persona_hash_a: str
    persona_hash_b: str
    scenario_id: str
    seed: int
    model: str


class SimCache(Protocol):
    def get(self, key: SimCacheKey) -> Transcript | None: ...

    def put(self, key: SimCacheKey, transcript: Transcript) -> None: ...


class InMemorySimCache:
    def __init__(self) -> None:
        self._data: dict[SimCacheKey, Transcript] = {}

    def get(self, key: SimCacheKey) -> Transcript | None:
        return self._data.get(key)

    def put(self, key: SimCacheKey, transcript: Transcript) -> None:
        self._data[key] = transcript

    def __len__(self) -> int:
        return len(self._data)


class JsonlSimCache:
    """One JSON object per line: ``{"key": {...}, "transcript": {...}}``. Loaded fully at
    construction; later lines for the same key win, so rewrites are plain appends."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._data: dict[SimCacheKey, Transcript] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    key = SimCacheKey(**row["key"])
                    self._data[key] = Transcript.model_validate(row["transcript"])

    def get(self, key: SimCacheKey) -> Transcript | None:
        return self._data.get(key)

    def put(self, key: SimCacheKey, transcript: Transcript) -> None:
        self._data[key] = transcript
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"key": asdict(key), "transcript": transcript.model_dump()}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self._data)
