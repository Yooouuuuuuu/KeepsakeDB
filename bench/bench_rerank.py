"""Does a reranker add anything on top of hybrid retrieval for memory recall?

Candidates come from KeepsakeDB (cached bge-m3 vectors, N distractors); each variant re-orders
them and we score the top-5. Variants:

  hybrid@15  -> top-5                      (what the store does today)
  hybrid@15  -> cross-encoder -> top-5     (bge-reranker-v2-m3 GGUF on llama-server, CPU)
  hybrid@30  -> cross-encoder -> top-5
  dense@15   -> cross-encoder -> top-5     ("skip hybrid, just rerank dense")
  bm25@15    -> cross-encoder -> top-5
  hybrid@15  -> LLM rerank -> top-5        (gemini flash via the OpenAI-compatible proxy)

    python bench/bench_rerank.py --n 10000 --rerank-url http://127.0.0.1:55603 --llm agy/gemini-3.8-flash
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(__file__))
from corpus import PROBES, documents  # noqa: E402

from keepsakedb import Keepsake  # noqa: E402
from keepsakedb.embedders import CachedEmbedder, OpenAIEmbedder  # noqa: E402
from keepsakedb.llms import OpenAIChat, extract_json  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "out")
K = 5


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def score(lists: List[List[int]]) -> Tuple[float, float, float]:
    r1 = r5 = mrr = 0.0
    for (q, ok), ids in zip(PROBES, lists):
        rank = next((i for i, d in enumerate(ids[:K]) if d in ok), None)
        if rank is not None:
            r5 += 1
            r1 += rank == 0
            mrr += 1 / (rank + 1)
    n = len(PROBES)
    return r1 / n, r5 / n, mrr / n


def cross_encoder(url: str, query: str, docs: List[str]) -> List[int]:
    if not docs:  # llama-server rejects an empty document list (400)
        return []
    body = json.dumps({"model": "reranker", "query": query, "documents": docs, "top_n": len(docs)},
                      ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url.rstrip("/") + "/v1/rerank", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        res = json.loads(r.read().decode())
    rows = res.get("results", res.get("data", []))
    rows = sorted(rows, key=lambda x: -(x.get("relevance_score", x.get("score", 0.0))))
    return [int(x["index"]) for x in rows]


LLM_SYSTEM = ("You rerank memories for an AI companion. Given the user's question and numbered candidate "
              "memories about the user, answer with JSON only: {\"ranking\": [numbers]} — the candidates most "
              "likely to contain the answer, best first, at most 5 numbers. If none fits, return the least bad.")


def llm_rerank(llm: OpenAIChat, query: str, docs: List[str]) -> List[int]:
    user = f"Question: {query}\n\nCandidates:\n" + "\n".join(f"{i + 1}. {d}" for i, d in enumerate(docs))
    try:
        ranking = extract_json(llm.complete(LLM_SYSTEM, user)).get("ranking", [])
        out = [int(x) - 1 for x in ranking if isinstance(x, (int, float, str)) and str(x).isdigit()]
        out = [i for i in out if 0 <= i < len(docs)]
    except Exception:
        out = []
    seen = set(out)
    return out + [i for i in range(len(docs)) if i not in seen]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--rerank-url", default="http://127.0.0.1:55603")
    ap.add_argument("--llm", default="agy/gemini-3.8-flash")
    ap.add_argument("--llm-url", default="http://127.0.0.1:18765/v1")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--skip-ce", action="store_true")
    args = ap.parse_args()

    emb = CachedEmbedder(OpenAIEmbedder("http://127.0.0.1:55602/v1", "bge-m3"), os.path.join(OUT, "emb_cache.sqlite"))
    docs = documents(args.n)
    vecs, qvecs = emb.embed(docs), emb.embed([q for q, _ in PROBES])
    assert emb.misses == 0, "vectors must be cached (run bench_run.py first)"
    path = os.path.join(OUT, "rerank.db")
    for ext in ("", "-wal", "-shm"):
        if os.path.exists(path + ext):
            os.remove(path + ext)
    ks = Keepsake(path, dims=vecs.shape[1])
    for i in range(0, len(docs), 1000):
        ks.add_many([(d, [], {}) for d in docs[i:i + 1000]], embeddings=vecs[i:i + 1000])

    # candidate lists (ids are 1-based rowids = doc index + 1)
    def cands(mode: str, n: int) -> List[List[int]]:
        return [[h.id - 1 for h in ks.search(q, k=n, mode=mode, query_embedding=qv, legs_k=max(n, 15))]
                for (q, _), qv in zip(PROBES, qvecs)]

    base = {"hybrid@15": cands("hybrid", 15), "hybrid@30": cands("hybrid", 30),
            "dense@15": cands("dense", 15), "bm25@15": cands("bm25", 15)}
    results = []
    for name in ("hybrid@15", "dense@15", "bm25@15"):
        results.append((f"{name} -> top-5 (no rerank)", score(base[name]), 0.0, 0.0))

    if not args.skip_ce:
        for name in ("hybrid@15", "hybrid@30", "dense@15", "bm25@15"):
            lists, lat = [], []
            for (q, _), ids in zip(PROBES, base[name]):
                texts = [docs[i] for i in ids]
                t0 = time.perf_counter()
                order = cross_encoder(args.rerank_url, q, texts)
                lat.append((time.perf_counter() - t0) * 1000)
                lists.append([ids[i] for i in order])
            results.append((f"{name} -> cross-encoder -> top-5", score(lists), pct(lat, 50), pct(lat, 95)))
            print(f"  done {name} cross-encoder: p50 {pct(lat, 50):.0f} ms", flush=True)

    if not args.skip_llm:
        llm = OpenAIChat(args.llm_url, args.llm, api_key="local-proxy-no-key")
        lists, lat = [], []
        for (q, _), ids in zip(PROBES, base["hybrid@15"]):
            texts = [docs[i] for i in ids]
            t0 = time.perf_counter()
            order = llm_rerank(llm, q, texts)
            lat.append((time.perf_counter() - t0) * 1000)
            lists.append([ids[i] for i in order])
        results.append((f"hybrid@15 -> LLM rerank ({args.llm}) -> top-5", score(lists), pct(lat, 50), pct(lat, 95)))

    print(f"\n=== {len(docs)} docs ===")
    for name, (r1, r5, mrr), p50, p95 in results:
        print(f"  {name:52s} R@1 {r1:.3f}  R@5 {r5:.3f}  MRR {mrr:.3f}  | rerank p50 {p50:7.0f} ms  p95 {p95:7.0f} ms")
    with open(os.path.join(OUT, f"rerank_{len(docs)}.json"), "w", encoding="utf-8") as f:
        json.dump([{"variant": n, "r1": s[0], "r5": s[1], "mrr": s[2], "p50_ms": p50, "p95_ms": p95}
                   for n, s, p50, p95 in results], f, ensure_ascii=False, indent=1)
    ks.close()


if __name__ == "__main__":
    main()
