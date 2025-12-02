# qepc/core/scripts.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

@dataclass
class GameScript:
    name: str
    pace_factor: float       # how much to scale pace
    offense_boost: float     # scalar added to ORtg
    volatility_factor: float # scale on volatility


# A simple 3-script universe. You can tune these later.
SCRIPTS: Dict[str, GameScript] = {
    "Grind": GameScript("Grind", pace_factor=0.94, offense_boost=-1.5, volatility_factor=0.9),
    "Balanced": GameScript("Balanced", pace_factor=1.0, offense_boost=0.0, volatility_factor=1.0),
    "Chaos": GameScript("Chaos", pace_factor=1.06, offense_boost=1.5, volatility_factor=1.1),
}

def default_script_probs() -> Dict[str, float]:
    # global priors; later this can depend on teams / rest / injuries
    return {"Grind": 0.25, "Balanced": 0.5, "Chaos": 0.25}

def sample_script(probs: Dict[str, float]) -> str:
    names = list(probs.keys())
    p = np.array([probs[n] for n in names])
    p = p / p.sum()
    return np.random.choice(names, p=p)
