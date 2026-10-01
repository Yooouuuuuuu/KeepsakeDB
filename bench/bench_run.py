"""KeepsakeDB vs Milvus Lite: quality (recall@k / MRR on labelled zh-TW probes buried in N
distractors) and speed (insert, query p50/p95, open time, file size).

    python bench/bench_run.py --n 0 1000 10000 --engines keepsake-bigram keepsake-jieba keepsake-vec milvus

Both engines get the same bge-m3 vectors (cached in bench/out/emb_cache.sqlite).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from corpus import PROBES, documents  # noqa: E402

from keepsakedb import Keepsake  # noqa: E402
from keepsakedb.embedders import CachedEmbedder, OpenAIEmbedder  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
K = 5
LEG_K = 15  # over-fetch per leg (3x) for hybrid, both engines


def pct(xs: List[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def score(results: Dict[str, List[int]]) -> Dict[str, float]:
    r1 = r5 = 0
    mrr = 0.0
    for q, ok in PROBES:
        ids = results[q]
        rank = next((i for i, d in enumerate(ids) if d in ok), None)
        if rank is not None:
            mrr += 1.0 / (rank + 1)
            r5 += 1
            r1 += rank == 0
    n = len(PROBES)
    return {"recall@1": r1 / n, "recall@5": r5 / n, "mrr": mrr / n}


# ------------------------------------------------------------------ engines
class KeepsakeEngine:
    def __init__(self, name: str, tokenizer: str, dense_index: str) -> None:
        self.name = name
        self.tokenizer = tokenizer
        self.dense_index = dense_index
        self.path = os.path.join(OUT, f"{name}.db")

    def build(self, docs: List[str], vecs: np.ndarray) -> float:
        for p in (self.path, self.path + "-wal", self.path + "-shm"):
            if os.path.exists(p):
                os.remove(p)
        t0 = time.perf_counter()
        ks = Keepsake(self.path, tokenizer=self.tokenizer, dense_index=self.dense_index, dims=vecs.shape[1])
        B = 1000
        for i in range(0, len(docs), B):
            ks.add_many([(d, [], {}) for d in docs[i:i + B]], embeddings=vecs[i:i + B])
        ks.optimize()
        ks.close()
        dt = time.perf_counter() - t0
        t1 = time.perf_counter()
        self.ks = Keepsake(self.path, tokenizer=self.tokenizer, dense_index=self.dense_index)
        self.open_s = time.perf_counter() - t1
        return dt

    def query(self, mode: str, q: str, qv: np.ndarray) -> List[int]:
        if mode == "hybrid-w":
            hits = self.ks.search(q, k=K, mode="hybrid", ranker="weighted", weights=(0.7, 0.3),
                                  query_embedding=qv, legs_k=LEG_K)
        else:
            hits = self.ks.search(q, k=K, mode=mode, query_embedding=qv, legs_k=LEG_K)
        return [h.id - 1 for h in hits]  # ids are 1-based rowids in insertion order

    def size_bytes(self) -> int:
        self.ks.optimize()
        return os.path.getsize(self.path)

    def close(self) -> None:
        self.ks.close()


class MilvusLiteEngine:
    name = "milvus-lite"

    def __init__(self) -> None:
        self.path = os.path.join(OUT, f"milvus_{int(time.time())}.db")

    def build(self, docs: List[str], vecs: np.ndarray) -> float:
        from pymilvus import DataType, Function, FunctionType, MilvusClient

        t0 = time.perf_counter()
        c = MilvusClient(self.path)
        s = c.create_schema()
        s.add_field("id", DataType.INT64, is_primary=True)
        s.add_field("text", DataType.VARCHAR, max_length=4000, enable_analyzer=True,
                    analyzer_params={"tokenizer": "jieba"})
        s.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        s.add_field("dense", DataType.FLOAT_VECTOR, dim=int(vecs.shape[1]))
        s.add_function(Function(name="bm25", function_type=FunctionType.BM25,
                                input_field_names=["text"], output_field_names=["sparse"]))
        ip = c.prepare_index_params()
        ip.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
        ip.add_index(field_name="dense", index_type="FLAT", metric_type="COSINE")
        c.create_collection("mem", schema=s, index_params=ip)
        B = 1000
        for i in range(0, len(docs), B):
            c.insert("mem", [{"id": i + j, "text": d, "dense": vecs[i + j].tolist()}
                             for j, d in enumerate(docs[i:i + B])])
        c.flush("mem") if hasattr(c, "flush") else None
        dt = time.perf_counter() - t0
        c.close()
        t1 = time.perf_counter()
        self.c = MilvusClient(self.path)
        self.open_s = time.perf_counter() - t1
        return dt

    def query(self, mode: str, q: str, qv: np.ndarray) -> List[int]:
        from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker

        if mode == "dense":
            r = self.c.search("mem", data=[qv.tolist()], anns_field="dense", limit=K)
        elif mode == "bm25":
            r = self.c.search("mem", data=[q], anns_field="sparse", limit=K)
        else:
            ranker = RRFRanker(60) if mode == "hybrid" else WeightedRanker(0.7, 0.3)
            r = self.c.hybrid_search("mem", [
                AnnSearchRequest(data=[qv.tolist()], anns_field="dense", param={}, limit=LEG_K),
                AnnSearchRequest(data=[q], anns_field="sparse", param={}, limit=LEG_K),
            ], ranker=ranker, limit=K)
        return [int(h["id"]) for h in r[0]]

    def size_bytes(self) -> int:
        total = 0
        for root, _, files in os.walk(self.path) if os.path.isdir(self.path) else []:
            total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
        return total or (os.path.getsize(self.path) if os.path.exists(self.path) else 0)

    def close(self) -> None:
        self.c.close()
        shutil.rmtree(self.path, ignore_errors=True)


ENGINES = {
    "keepsake-bigram": lambda: KeepsakeEngine("keepsake-bigram", "bigram", "numpy"),
    "keepsake-jieba": lambda: KeepsakeEngine("keepsake-jieba", "jieba", "numpy"),
    "keepsake-vec": lambda: KeepsakeEngine("keepsake-vec", "bigram", "sqlite-vec"),
    "milvus": MilvusLiteEngine,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[0, 1000, 10000])
    ap.add_argument("--engines", nargs="+", default=list(ENGINES))
    ap.add_argument("--modes", nargs="+", default=["dense", "bm25", "hybrid", "hybrid-w"])
    ap.add_argument("--embedder", default="http://127.0.0.1:55602/v1")
    ap.add_argument("--repeat", type=int, default=3, help="query passes for latency")
    args = ap.parse_args()

    emb = CachedEmbedder(OpenAIEmbedder(args.embedder, "bge-m3", batch_size=64),
                         os.path.join(OUT, "emb_cache.sqlite"))
    all_results = []
    for n in args.n:
        docs = documents(n)
        t0 = time.perf_counter()
        vecs = emb.embed(docs)
        qvecs = emb.embed([q for q, _ in PROBES])
        print(f"\n=== N = {len(docs)} docs ({n} distractors) | embeddings {time.perf_counter() - t0:.1f}s "
              f"(cache hits {emb.hits}, misses {emb.misses}) ===", flush=True)
        for ename in args.engines:
            eng = ENGINES[ename]()
            build_s = eng.build(docs, vecs)
            row = {"n_docs": len(docs), "engine": ename, "build_s": round(build_s, 2),
                   "open_s": round(eng.open_s, 3), "size_mb": round(eng.size_bytes() / 1e6, 2)}
            for mode in args.modes:
                lat: List[float] = []
                results: Dict[str, List[int]] = {}
                for rep in range(args.repeat):
                    for (q, _), qv in zip(PROBES, qvecs):
                        t1 = time.perf_counter()
                        ids = eng.query(mode, q, qv)
                        lat.append((time.perf_counter() - t1) * 1000)
                        results[q] = ids
                sc = score(results)
                row[mode] = {**{k: round(v, 3) for k, v in sc.items()},
                             "p50_ms": round(pct(lat, 50), 2), "p95_ms": round(pct(lat, 95), 2)}
                print(f"  {ename:16s} {mode:7s} R@1 {sc['recall@1']:.2f} R@5 {sc['recall@5']:.2f} "
                      f"MRR {sc['mrr']:.3f} | p50 {pct(lat, 50):6.2f} ms  p95 {pct(lat, 95):6.2f} ms", flush=True)
                if mode == "hybrid" and n == args.n[-1]:
                    misses = [(q, ok, results[q]) for q, ok in PROBES if not any(d in ok for d in results[q])]
                    row["hybrid_misses"] = [{"q": q, "expected": ok, "got": got} for q, ok, got in misses]
            print(f"  {ename:16s} build {build_s:.2f}s  open {eng.open_s * 1000:.0f} ms  size {row['size_mb']} MB", flush=True)
            eng.close()
            all_results.append(row)
        with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=1)
    print("\nwrote", os.path.join(OUT, "results.json"))


if __name__ == "__main__":
    main()
