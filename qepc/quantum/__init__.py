"""
QEPC quantum-inspired utilities.

This package holds small, focused tools that implement the
"quantum-flavored" design ideas:

- decoherence: recency weighting with stat-specific time constants
- entropy: uncertainty/confidence measures for simulated distributions
- (later) entanglement: correlation / co-movement structure
"""

from . import decoherence
from . import entropy

__all__ = ["decoherence", "entropy"]
