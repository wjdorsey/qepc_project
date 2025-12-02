# qepc/config.py

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass

@dataclass
class QEPCConfig:
    project_root: Path
    data_root: Path
    raw_root: Path
    cache_root: Path
    seed: int = 42

    @classmethod
    def from_project_root(cls, root: Path) -> "QEPCConfig":
        data = root / "data"
        return cls(
            project_root=root,
            data_root=data,
            raw_root=data / "raw",
            cache_root=data / "cache",
            seed=42,
        )

def detect_project_root() -> Path:
    """Walk upwards until we see a 'data' folder or .git; fallback to cwd."""
    cur = Path.cwd()
    for _ in range(6):
        if (cur / "data").exists() or (cur / ".git").exists():
            return cur
        cur = cur.parent
    return Path.cwd()  # fallback
