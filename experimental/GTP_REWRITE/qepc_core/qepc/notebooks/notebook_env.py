# qepc/notebooks/notebook_env.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from qepc.config import QEPCConfig, detect_project_root
from qepc.logging_utils import qstep

@dataclass
class NotebookEnv:
    project_root: Path
    config: QEPCConfig

def init_notebook_env() -> NotebookEnv:
    root = detect_project_root()
    cfg = QEPCConfig.from_project_root(root)
    qstep(f"Notebook env initialized – project_root={root}")
    return NotebookEnv(project_root=root, config=cfg)
