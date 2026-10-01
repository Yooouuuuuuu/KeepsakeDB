"""Fusion sweep on cached vectors: RRF k, weighted (arctan / min-max) dense share, weighted RRF.

    python bench/sweep_fusion.py --n 1000 10000
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from corpus import PROBES, documents  # noqa: E402

from keepsakedb import Keepsake  # noqa: E402
from keepsakedb.embedders import CachedEmbedder, OpenAIEmbedder  # noqa: E402
from keepsakedb.store import fusion  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "out")
K, LEG_K = 5, 15


def score(lists):
    r5 = mrr = 0.0
    for (q, ok), ids in zip(PROBES, lists):
        rank = next((i for i, d in enumerate(ids) if d in ok), None)
        if rank is not None:
            r5 += 1
            mrr += 1 / (rank + 1)
    return r5 / len(PROBES), mrr / len(PROBES)


def wrrf(legs, weights, k):
    out = {}
    for leg, w in zip(legs, weights):
        for rank, (mid, _) in enumerate(leg, start=1):
            out[mid] = out.get(mid, 0.0) + w / (k + rank)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[1000, 10000])
    ap.add_argument("--leg_k", type=int, nargs="+", default=[15])
    args = ap.parse_args()
    emb = CachedEmbedder(OpenAIEmbedder("http://127.0.0.1:55602/v1", "bge-m3"), os.path.join(OUT, "emb_cache.sqlite"))
    for n in args.n:
        docs = documents(n)
        vecs, qvecs = emb.embed(docs), emb.embed([q for q, _ in PROBES])
        assert emb.misses == 0, "vectors must be cached (run bench_run.py first)"
        path = os.path.join(OUT, "sweep.db")
        for ext in ("", "-wal", "-shm"):
            if os.path.exists(path + ext):
                os.remove(path + ext)
        ks = Keepsake(path, dims=vecs.shape[1])
        for i in range(0, len(docs), 1000):
            ks.add_many([(d, [], {}) for d in docs[i:i + 1000]], embeddings=vecs[i:i + 1000])
        for leg_k in args.leg_k:
            legs = [(ks._dense_leg(q, leg_k, False, None, qv), ks._bm25_leg(q, leg_k, False, None))
                    for (q, _), qv in zip(PROBES, qvecs)]
            print(f"\n=== {len(docs)} docs, over-fetch {leg_k} per leg ===")
            rows = []
            rows.append(("dense only", score([[m - 1 for m, _ in d[:K]] for d, _ in legs])))
            rows.append(("bm25 only", score([[m - 1 for m, _ in b[:K]] for _, b in legs])))
            for k in (10, 30, 60, 100):
                rows.append((f"RRF k={k}", score([[m - 1 for m, _ in fusion.top(fusion.rrf([d, b], k), K)] for d, b in legs])))
            for wd in (0.5, 0.6, 0.7, 0.8, 0.9):
                for norm in ("arctan", "minmax"):
                    rows.append((f"weighted {norm} dense={wd:.1f}", score(
                        [[m - 1 for m, _ in fusion.top(fusion.weighted([d, b], (wd, 1 - wd), norm), K)] for d, b in legs])))
            for wd in (0.5, 0.6, 0.7, 0.8):
                rows.append((f"weighted-RRF k=60 dense={wd:.1f}", score(
                    [[m - 1 for m, _ in fusion.top(wrrf([d, b], (wd, 1 - wd), 60), K)] for d, b in legs])))
            for name, (r5, mrr) in rows:
                print(f"  {name:34s} R@5 {r5:.3f}  MRR {mrr:.3f}")
        ks.close()


if __name__ == "__main__":
    main()
