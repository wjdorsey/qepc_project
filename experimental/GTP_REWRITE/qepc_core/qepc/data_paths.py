# qepc/logging_utils.py

from __future__ import annotations
from datetime import datetime

def qstep(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] ⧉ QEPC: {msg}")

def qwarn(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] ⚠️ QEPC WARN: {msg}")

def qerr(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] ❌ QEPC ERROR: {msg}")
