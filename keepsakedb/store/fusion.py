"""Rank fusion for the hybrid search — the two rankers Milvus ships.

RRF:      score(d) = sum_i 1 / (k + rank_i(d)), rank 1-based, k = 60 by default
          (Milvus RRFRanker default; the docs recommend k in [10, 100]).
Weighted: score(d) = sum_i w_i * norm_i(score_i(d)).
          Milvus Standalone normalizes with arctan (unbounded BM25 -> [0, 1));
          Milvus Lite 3.2 uses per-query min-max. Both are available here; min-max is the
          default because it won the fusion sweep (docs/BENCHMARK.md): arctan squashes
          BM25 scores (typically 5-15) into a flat 0.9-0.98 band, so the BM25 leg stops
          discriminating.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

Leg = Sequence[Tuple[int, float]]  # (id, score) sorted best-first


def rrf(legs: Sequence[Leg], k: float = 60.0) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for leg in legs:
        for rank, (mid, _score) in enumerate(leg, start=1):
            out[mid] = out.get(mid, 0.0) + 1.0 / (k + rank)
    return out


def weighted(legs: Sequence[Leg], weights: Sequence[float], norm: str = "minmax") -> Dict[int, float]:
    if len(weights) != len(legs):
        raise ValueError("one weight per leg")
    out: Dict[int, float] = {}
    for leg, w in zip(legs, weights):
        if not leg:
            continue
        scores = [s for _, s in leg]
        if norm == "minmax":
            lo, hi = min(scores), max(scores)
            rng = hi - lo
            normed = [(s - lo) / rng if rng > 0 else 1.0 for s in scores]
        elif norm == "arctan":
            normed = [math.atan(max(s, 0.0)) * 2.0 / math.pi for s in scores]
        elif norm == "none":
            normed = scores
        else:
            raise ValueError(f"unknown norm {norm!r}")
        for (mid, _), n in zip(leg, normed):
            out[mid] = out.get(mid, 0.0) + w * n
    return out


def ranks(leg: Leg) -> Dict[int, int]:
    return {mid: r for r, (mid, _) in enumerate(leg, start=1)}


def top(scored: Dict[int, float], k: int) -> List[Tuple[int, float]]:
    return sorted(scored.items(), key=lambda x: (-x[1], x[0]))[:k]
