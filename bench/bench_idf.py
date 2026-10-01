"""The IDF experiment: does segment-local BM25 (Milvus Lite) drift from global BM25?

Insert the labelled facts + N distractors (N = 30000 -> three memtable flushes in Lite),
then compare each engine's BM25 top-10 with a global-statistics oracle that uses the SAME
tokens, k1, b and IDF formula as that engine. Reports top-10 overlap, top-1 agreement and
the mean rank of the oracle's #1 inside the engine's list.

    python bench/bench_idf.py --n 30000
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.dirname(__file__))
from corpus import PROBES, documents  # noqa: E402

from keepsakedb import Keepsake  # noqa: E402
from keepsakedb.store.tokenize import JiebaTokenizer, BigramTokenizer, index_string  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
TOPK = 10


class GlobalBM25:
    """Exact BM25 with global statistics; `variant` selects the IDF formula."""

    def __init__(self, docs_tokens: List[List[str]], k1: float, b: float, variant: str) -> None:
        self.k1, self.b, self.variant = k1, b, variant
        self.N = len(docs_tokens)
        self.dl = [len(t) for t in docs_tokens]
        self.avgdl = sum(self.dl) / max(self.N, 1)
        self.df: Counter = Counter()
        self.post: Dict[str, List[tuple]] = defaultdict(list)
        for i, toks in enumerate(docs_tokens):
            c = Counter(toks)
            for t, tf in c.items():
                self.df[t] += 1
                self.post[t].append((i, tf))

    def idf(self, t: str) -> float:
        n = self.df[t]
        if self.variant == "lite":   # milvus-lite: log((N-n+0.5)/(n+0.5)+1)
            return math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)
        # fts5: log((N-n+0.5)/(n+0.5)), floored at a small positive value like FTS5 does
        return max(math.log((self.N - n + 0.5) / (n + 0.5)), 1e-6)

    def top(self, q_tokens: List[str], k: int) -> List[int]:
        scores: Dict[int, float] = defaultdict(float)
        for t in set(q_tokens):
            if t not in self.post:
                continue
            idf = self.idf(t)
            for i, tf in self.post[t]:
                scores[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
        return [i for i, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]]


def compare(name: str, engine_lists: Dict[str, List[int]], oracle_lists: Dict[str, List[int]]) -> None:
    overlaps, top1, ranks = [], 0, []
    for q, _ in PROBES:
        e, o = engine_lists[q], oracle_lists[q]
        if not o:
            continue
        overlaps.append(len(set(e[:TOPK]) & set(o[:TOPK])) / TOPK)
        top1 += bool(e) and e[0] == o[0]
        ranks.append(e.index(o[0]) + 1 if o[0] in e else TOPK + 1)
    print(f"  {name:28s} top-10 overlap {sum(overlaps) / len(overlaps):.3f} | top-1 agree {top1}/{len(overlaps)} "
          f"| oracle #1 at mean rank {sum(ranks) / len(ranks):.2f} (11 = not in top-10)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    args = ap.parse_args()
    docs = documents(args.n)
    print(f"docs: {len(docs)}")

    # ---- Milvus Lite (jieba) vs oracle(lite formula, k1=1.5, b=0.75, jieba tokens)
    from pymilvus import DataType, Function, FunctionType, MilvusClient

    path = os.path.join(OUT, f"idf_{int(time.time())}.db")
    c = MilvusClient(path)
    s = c.create_schema()
    s.add_field("id", DataType.INT64, is_primary=True)
    s.add_field("text", DataType.VARCHAR, max_length=4000, enable_analyzer=True, analyzer_params={"tokenizer": "jieba"})
    s.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    s.add_field("dense", DataType.FLOAT_VECTOR, dim=4)
    s.add_function(Function(name="bm25", function_type=FunctionType.BM25, input_field_names=["text"], output_field_names=["sparse"]))
    ip = c.prepare_index_params()
    ip.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    ip.add_index(field_name="dense", index_type="FLAT", metric_type="COSINE")
    c.create_collection("mem", schema=s, index_params=ip)
    t0 = time.perf_counter()
    for i in range(0, len(docs), 1000):
        c.insert("mem", [{"id": i + j, "text": d, "dense": [0.1, 0.2, 0.3, 0.4]} for j, d in enumerate(docs[i:i + 1000])])
    print(f"milvus-lite insert {time.perf_counter() - t0:.1f}s")
    lite_lists, lat = {}, []
    for q, _ in PROBES:
        t1 = time.perf_counter()
        r = c.search("mem", data=[q], anns_field="sparse", limit=TOPK)
        lat.append((time.perf_counter() - t1) * 1000)
        lite_lists[q] = [int(h["id"]) for h in r[0]]
    print(f"milvus-lite bm25 query mean {sum(lat) / len(lat):.1f} ms")
    c.close()

    # same tokens as Lite: jieba cut_for_search, lowercased
    jt = JiebaTokenizer()
    jtoks = [[u[0] for u in jt.units(d)] for d in docs]
    oracle_lite = GlobalBM25(jtoks, 1.5, 0.75, "lite")
    o_lists = {q: oracle_lite.top([u[0] for u in jt.units(q)], TOPK) for q, _ in PROBES}
    print("\nBM25 ranking vs a GLOBAL-statistics oracle with the engine's own tokens/k1/b/formula:")
    compare("milvus-lite (jieba)", lite_lists, o_lists)

    # ---- KeepsakeDB FTS5 (jieba and bigram) vs oracle(fts5 formula, k1=1.2, b=0.75)
    for tok_name, tok in (("jieba", jt), ("bigram", BigramTokenizer())):
        p = os.path.join(OUT, f"idf_{tok_name}.db")
        for ext in ("", "-wal", "-shm"):
            if os.path.exists(p + ext):
                os.remove(p + ext)
        ks = Keepsake(p, tokenizer=tok_name)
        t0 = time.perf_counter()
        for i in range(0, len(docs), 1000):
            ks.add_many([(d, [], {}) for d in docs[i:i + 1000]])
        ks.optimize()
        print(f"\nkeepsake ({tok_name}) insert {time.perf_counter() - t0:.1f}s")
        k_lists, lat = {}, []
        for q, _ in PROBES:
            t1 = time.perf_counter()
            k_lists[q] = [h.id - 1 for h in ks.search(q, k=TOPK, mode="bm25")]
            lat.append((time.perf_counter() - t1) * 1000)
        print(f"keepsake ({tok_name}) bm25 query mean {sum(lat) / len(lat):.1f} ms")
        toks = [index_string(tok, d).split() for d in docs]
        oracle = GlobalBM25(toks, 1.2, 0.75, "fts5")
        o2 = {q: oracle.top(index_string(tok, q).split(), TOPK) for q, _ in PROBES}
        compare(f"keepsake fts5 ({tok_name})", k_lists, o2)
        ks.close()


if __name__ == "__main__":
    main()
