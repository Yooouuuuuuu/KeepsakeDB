"""KeepsakeDB — a lightweight hybrid (dense + BM25) memory store for companions and agents.

L1 ``keepsakedb.store``     the store (this is the DB; usable alone)
L2 ``keepsakedb.pipeline``  the memory pipeline (extract / tag / merge / archive) — pluggable
L3 ``keepsakedb.integrate`` how an agent wires L1+L2 into its turn loop (any framework)
"""
from .oplog import configure_logging
from .store import Keepsake, Memory

__version__ = "0.1.3"
__all__ = ["Keepsake", "Memory", "configure_logging", "__version__"]
