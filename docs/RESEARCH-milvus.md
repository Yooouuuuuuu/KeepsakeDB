# Research: can SQLite do the hybrid mix? — what Milvus does, what Milvus Lite cannot, and the trial design

Date: 2026-10-01. Everything below is verified against the installed packages and the
official docs on that date; file references are to `milvus_lite` 3.2.1 as installed from PyPI.

## 1. The question

The 556 memory design wants Milvus-style hybrid retrieval — dense vectors **and** BM25 keyword
search, fused — but embedded in one process, one file, no Docker, on Windows. Milvus proper
needs a service. The earlier note in 556's `MEMORY-DESIGN.md` said Milvus Lite "has no
sparse/BM25/hybrid and no native Windows support". That was true for the 2.4.x generation and
is **no longer true**; the reasons it still falls short are more specific and more instructive.

## 2. Milvus Lite today (3.2.1, released 2026-08-25)

Facts from the README and the source:

- **Rewritten in pure Python.** "Instead of wrapping the full C++ Milvus core, it uses a local
  LSM-style storage engine with WAL, in-memory memtables, immutable Parquet segments,
  segment-level FAISS indexes, scalar filtering, BM25 full text search, and a
  Milvus-compatible gRPC adapter." Wheel is `py3-none-any`; deps are faiss-cpu, pyarrow,
  grpcio, numpy (+ jieba optional).
- **Windows works.** CI covers Windows on Python 3.10. Note `pymilvus[milvus-lite]` excludes
  win32 in its extra, so on Windows you `pip install milvus-lite` directly. Verified here:
  `MilvusClient("file.db")` starts in 0.18 s, BM25 with jieba and `hybrid_search` with
  `RRFRanker` return correct results (`bench/smoke_milvus_lite.py`).
- **It has the features**: BM25 `Function` → `SPARSE_FLOAT_VECTOR`, `SPARSE_INVERTED_INDEX`,
  `hybrid_search` with `RRFRanker` / `WeightedRanker`, analyzers `standard` and `jieba`
  (the Standalone name `chinese` is not accepted).

### 2.1 Why it still cannot do the mix *well* — the segment-local IDF problem

`index/sparse_inverted.py` (lines 1–15): *"SparseInvertedIndex — per-segment BM25 inverted
index … evaluates BM25 scoring at query time using segment-local statistics."*

`engine/collection.py`, `_search_sparse` docstring:

> Each immutable segment gets a cached SparseInvertedIndex … The mutable memtable's index is
> rebuilt each search … Per-source top-k results are merged globally.
> **TODO: IDF accuracy — each segment currently uses its own IDF statistics, so BM25 scores
> from different segments have different baselines.** Fix: aggregate global statistics
> (doc_count/avgdl/df summed across segments) at search time and use global IDF for scoring.
> Similar to Elasticsearch's DFS_QUERY_THEN_FETCH strategy.

What this means mechanically (`constants.py`: `MEMTABLE_SIZE_LIMIT = 10_000`,
`MAX_SEGMENT_ROWS = 100_000`):

1. Rows go WAL → memtable. Every 10,000 rows the memtable is flushed into an immutable segment.
2. Each segment computes its own `N`, `df`, `avgdl`, so `IDF(term) = log((N-df+0.5)/(df+0.5)+1)`
   differs per segment. A rare term in one segment and a common one in another get
   incomparable scores; the merge just sorts raw scores.
3. The memtable's inverted index is **rebuilt on every search** (a Python loop over all
   unflushed rows and terms), so query latency grows with the memtable up to the flush.
4. Scoring is term-at-a-time in Python dicts (`sparse_inverted.py` lines 141–157): no WAND /
   MaxScore pruning, no C. Fine for thousands of rows, not a search engine.

Below 10,000 memories there is a single memtable, so the IDF problem does not surface — but
the companion store is written continuously and will cross flush boundaries over time, and
a restart replays the WAL into a fresh memtable while old rows live in segments.

Other Lite-specific constraints: single process per `data_dir` (file lock), writers must be
serialized, BM25 default `k1 = 1.5` (Standalone: 1.2), `WeightedRanker` normalizes with
per-query **min-max** (`adapter/grpc/reranker.py`) whereas Standalone documents **arctan**.

### 2.2 What stayed the same as Standalone (the model worth copying)

- Analyzer → term ids (FNV-1a, `analyzer/hash.py`) → term frequencies counted at insert
  (`analyzer/sparse.py: compute_tf`) stored as the sparse vector. The query text is passed
  through the same analyzer. (`standard` is a `\w+` regex: a CJK run becomes one token,
  so Chinese needs `jieba`.)
- `RRFRanker`: `score = Σ 1/(k + rank)`, `k = 60` default (`adapter/grpc/reranker.py`
  line 186; Standalone docs: k in (0, 16384), recommended [10, 100]). Each sub-request has
  its own `limit` (over-fetch), the ranker re-sorts, then the final `limit` applies.

## 3. Milvus Standalone's hybrid (the reference)

From the official docs (`full-text-search`, `sparse-inverted-index`, `rrf-ranker`,
`weighted-ranker`, analyzer pages):

| Piece | Standalone |
|---|---|
| Text → sparse | `Function(type=BM25)` over a `VARCHAR` with `enable_analyzer=True`; output `SPARSE_FLOAT_VECTOR`, not readable in `output_fields` |
| Analyzer | `standard`, `chinese` (jieba), `multi-language` (language identifier), filters (lowercase, stop words, pinyin…) — Tantivy-based (Rust) inside segcore via a C++/Rust bridge |
| Index | `SPARSE_INVERTED_INDEX`, `metric_type="BM25"`, `bm25_k1=1.2`, `bm25_b=0.75`, `inverted_index_algo` = `DAAT_MAXSCORE` (default) / `DAAT_WAND` / `TAAT_NAIVE` / `BLOCK_MAX_*`; `drop_ratio_search` prunes low-weight query terms |
| Statistics | **global** collection statistics at query time (the docs describe `TAAT_NAIVE` as the baseline "when you need scoring to adapt dynamically to global collection statistics"; DAAT variants pre-compute per-term upper bounds) |
| Fusion | `RRFRanker(k=60)`; `WeightedRanker(w…)` with arctan normalization to [0,1], weights in [0,1], `norm_score` flag |
| Deployment | a server (Docker/binary), gRPC, etcd + object storage (Standalone bundles them) |

So "Milvus's hybrid" is: **one global BM25 index + one dense index, each queried with
over-fetch, fused by rank.** Nothing in that model requires Milvus.

## 4. The SQLite trial: mapping Milvus onto FTS5 + vectors

| Milvus | KeepsakeDB on SQLite |
|---|---|
| analyzer (standard / jieba) | Python pre-tokenizer (`store/tokenize.py`): **bigram** (default, zero deps, same semantics as Lucene CJKAnalyzer and Hermes's `cjk_unicode61`) or **jieba**; FTS5 indexes the token string with `unicode61` |
| BM25 Function + SPARSE_INVERTED_INDEX | FTS5 inverted index + `bm25()` rank function (k1 = 1.2, b = 0.75 — Standalone's defaults), **global statistics by construction** (one index per table), C implementation, incremental updates, no flush/segment logic |
| FLAT dense index | exact cosine over a normalized numpy matrix (default) or a `sqlite-vec` `vec0` table in the same file |
| RRFRanker / WeightedRanker | `store/fusion.py`: RRF (k = 60) / weighted with arctan (Standalone) or min-max (Lite) |
| per-request `limit` over-fetch → final `limit` | `oversample` (default 3×) per leg → final `k` |
| partition / `user_id` isolation | one `.db` file per companion |
| scalar filter expressions | SQL `WHERE` (archived state, tags) |

Why FTS5's built-ins are not enough (measured on SQLite 3.53.1): `unicode61` returns 0 hits for
罐頭 inside a sentence (the whole CJK run is one token) and `trigram` needs ≥ 3 characters, so
2-character words — most Chinese words — never match. Pre-tokenizing into bigrams and querying
them as an FTS5 **phrase** gives exact substring matching down to 2 characters, which is
exactly the Hermes `cjk_unicode61` trick without the native extension.

Where SQLite is genuinely weaker than Milvus Standalone (and we accept it at companion scale):

- no WAND/MaxScore pruning — FTS5 scores every posting of an OR query; irrelevant below ~10⁵ rows;
- no learned-sparse vectors (bge-m3 lexical weights / SPLADE) — would need our own term table;
  BM25 is the lexical leg for now;
- single writer; no distributed anything. Outgrow point ≈ 10⁶ memories → graduate to Milvus
  Standalone with the same conceptual model (the API shape is kept deliberately close).

Where SQLite is **better** for a memory store: global IDF always; one file, zero processes,
~0 startup; archive/tags/history are plain tables; incremental index updates with transactions.

**This is a trial, not a commitment.** The decision rule is the benchmark in section 5: if
SQLite loses on recall against Milvus Lite with the same embeddings, or cannot stay in the
low-millisecond range per query at 10⁴–10⁵ memories, we switch the L1 backend (the layered
design makes L1 replaceable) — candidates would be an in-process Milvus Lite behind the same
`Keepsake` API, or our own inverted index.

## 5. Benchmark plan (results in `docs/BENCHMARK.md`)

Same bge-m3 (1024-d, llama-server, CPU) embeddings for every engine, cached once.

1. **Quality** — the 556 endurance corpus (30 facts + 5 corrections in zh-TW, 20 paraphrased
   probes with labelled answers) buried in N synthetic companion-style distractors,
   N ∈ {0, 1k, 10k}: recall@1/@5 and MRR for dense-only, BM25-only, hybrid (RRF k=60) on
   KeepsakeDB (bigram, jieba) vs Milvus Lite (jieba, FLAT + BM25, RRF k=60).
2. **Speed** — insert throughput, query latency p50/p95 per mode, open time, file size, at
   each N.
3. **The IDF experiment** — insert 30k rows into Milvus Lite (three flushes) and compare its
   BM25 top-10 with a global-statistics oracle (`rank_bm25`, same tokens, same k1/b) and with
   FTS5: rank overlap and score drift. This isolates the segment-local effect.

## 6. Sources

- Milvus Lite README (github.com/milvus-io/milvus-lite): architecture, feature table,
  limitations ("BM25 IDF statistics are segment-local rather than global"), platforms.
- `milvus_lite` 3.2.1 source: `engine/collection.py`, `index/sparse_inverted.py`,
  `analyzer/{standard,jieba_analyzer,sparse,hash}.py`, `adapter/grpc/reranker.py`, `constants.py`.
- Milvus docs: Full Text Search, SPARSE_INVERTED_INDEX, RRF Ranker, Weighted Ranker,
  Chinese/Jieba analyzer, Milvus Lite vs Standalone vs Distributed.
- PyPI: `milvus-lite` 2.4.x wheels were manylinux/macOS only; 3.x is `py3-none-any`;
  `pymilvus` 3.0.2 extra `milvus-lite` is `sys_platform != "win32"`.
- Hermes Agent `hermes_state_fts.py`: the `cjk_unicode61` loadable tokenizer design note.
