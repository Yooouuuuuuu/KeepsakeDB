# Benchmark: KeepsakeDB (SQLite) vs Milvus Lite 3.2.1

Date 2026-10-01 · Windows 11, CPU only · Python 3.11 · SQLite 3.53.1 · milvus-lite 3.2.1 / pymilvus 3.0.2
Embedder: bge-m3 Q8_0 GGUF on llama-server (1024-d, CPU, ≈ 13.6 ms/text in batches of 64 — 10k
memories re-embed in ≈ 2.5 min). **Every engine receives the identical vectors**, so the dense-only
rows are a consistency check and every difference comes from the lexical leg and the fusion.

Scripts: `bench/corpus.py`, `bench/bench_run.py`, `bench/bench_idf.py`. Raw numbers: `bench/out/results.json`.

## Setup

- Corpus: 35 labelled zh-TW companion memories (30 facts + 5 corrections, from the 556 endurance
  test) + N synthetic companion-style distractors (third parties and non-conflicting first-person
  topics; high lexical/semantic overlap), N ∈ {0, 1 000, 10 000}.
- Probes: 40 paraphrased first-person questions with the acceptable memory ids. recall@k = an
  acceptable id in the top-k; MRR over the first acceptable hit. k = 5, over-fetch 15 per leg.
- Engines: `keepsake-bigram` (FTS5 + numpy), `keepsake-jieba` (FTS5 + numpy), `keepsake-vec`
  (FTS5 + sqlite-vec), `milvus-lite` (jieba analyzer, FLAT + SPARSE_INVERTED_INDEX/BM25, in-process).
- Fusion: RRF k = 60 (`hybrid`); weighted 0.7 dense / 0.3 BM25 (`hybrid-w`: arctan normalization
  in KeepsakeDB, `WeightedRanker` in Milvus Lite).
- Latency: 3 passes × 40 probes, p50 / p95 per query, embedding time excluded (vectors precomputed).

## Quality

| docs | engine | dense R@5 | BM25 R@5 | **hybrid R@5** | hybrid MRR | hybrid-w R@5 |
|---:|---|---:|---:|---:|---:|---:|
| 35 | keepsake-bigram | 0.97 | 0.82 | **0.95** | 0.888 | 0.97 |
| 35 | keepsake-jieba | 0.97 | 0.75 | 0.90 | 0.833 | 0.95 |
| 35 | milvus-lite | 0.97 | 0.78 | 0.90 | 0.826 | 0.90 |
| 1 035 | keepsake-bigram | 0.82 | 0.65 | **0.85** | 0.762 | 0.82 |
| 1 035 | keepsake-jieba | 0.82 | 0.55 | 0.75 | 0.704 | 0.80 |
| 1 035 | milvus-lite | 0.82 | 0.55 | 0.70 | 0.658 | 0.78 |
| 10 035 | keepsake-bigram | 0.70 | 0.68 | **0.88** | 0.813 | 0.80 |
| 10 035 | keepsake-jieba | 0.70 | 0.55 | 0.80 | 0.721 | 0.75 |
| 10 035 | milvus-lite | 0.70 | 0.53 | 0.78 | 0.725 | 0.72 |

(`keepsake-vec` equals `keepsake-bigram` on quality: same index, same vectors.)

What the numbers say:

1. **Hybrid earns its keep as the store grows.** At 10k docs it adds +0.18 recall@5 over dense-only
   (0.70 → 0.88). Dense alone degrades fastest with a crowded haystack; BM25 keeps names, places
   and numbers findable.
2. **Bigram beats jieba for Traditional Chinese** on every row (BM25 0.68 vs 0.55 at 10k). jieba's
   dictionary is Simplified-oriented and `cut_for_search` fragments zh-TW text; bigrams need no
   dictionary and give exact substring matching down to two characters. Milvus Lite cannot do
   this: its analyzers are `standard` (one token per CJK run) and `jieba`.
3. **KeepsakeDB's hybrid beats Milvus Lite's** at every size (0.88 vs 0.78 at 10k). Same vectors,
   same RRF — the gap is the lexical leg (bigram + global BM25 statistics).
4. **RRF is the robust default.** Weighted 0.7/0.3 wins only at 35 docs, where dense is already
   near-perfect, and loses at 10k. This matches Milvus's own guidance (RRF when you cannot say which
   leg should dominate).

Hybrid misses at 10k (keepsake-bigram, 5 of 40): 我媽媽住哪裡、做什麼的 (buried under 媽媽
distractors by design), 我現在住在哪個城市 (inferential: 新竹 is only implied), 我討厭什麼天氣,
我的貓叫什麼名字 (Mochi is a name the question cannot contain), 東京要去幾天. Milvus Lite misses
those plus 我的生日, 我在哪間公司上班, 我去日本要去多久, 我在公司哪個單位.

## Speed (per query, ms)

| docs | engine | dense p50 | BM25 p50 | **hybrid p50 (p95)** | build | open | file |
|---:|---|---:|---:|---:|---:|---:|---:|
| 35 | keepsake-bigram | 0.05 | 0.07 | **0.13** (0.16) | 0.08 s | 11 ms | 0.24 MB |
| 35 | milvus-lite | 1.41 | 1.41 | 2.27 (2.76) | 1.30 s | — | 0.37 MB |
| 1 035 | keepsake-bigram | 0.19 | 0.11 | **0.34** (0.44) | 0.24 s | 18 ms | 5.1 MB |
| 1 035 | keepsake-jieba | 0.17 | 0.49 | 0.79 (0.99) | 0.26 s | 34 ms | 5.1 MB |
| 1 035 | keepsake-vec | 5.25 | 0.13 | 5.93 (7.61) | 0.72 s | 9 ms | 13.6 MB |
| 1 035 | milvus-lite | 5.93 | 2.11 | 7.93 (8.98) | 1.50 s | — | 9.2 MB |
| 10 035 | keepsake-bigram | 0.66 | 0.84 | **1.97** (4.41) | 1.52 s | 78 ms | 49 MB |
| 10 035 | keepsake-jieba | 0.69 | 6.88 | 8.19 (9.84) | 2.01 s | 78 ms | 49 MB |
| 10 035 | keepsake-vec | 25.8 | 0.90 | 27.8 (31.5) | 6.46 s | 13 ms | 92 MB |
| 10 035 | milvus-lite | 43.3 | 10.4 | 53.1 (62.8) | 4.73 s | — | 83 MB |

(Milvus Lite "open" is a client reconnect to its still-running server, so it is not comparable;
"build" includes index build and, for Milvus, the BM25 analysis of every row.)

What the numbers say:

1. **KeepsakeDB answers a hybrid query in ≈ 2 ms at 10k memories; Milvus Lite needs ≈ 53 ms**
   (27×). Dense: 0.66 vs 43 ms — exact cosine over a numpy matrix against Milvus Lite's FLAT path
   through gRPC and a Python memtable. BM25: 0.84 vs 10.4 ms — FTS5 in C versus a per-segment
   Python loop.
2. **numpy scales linearly** (≈ 0.06 ms per 1 000 memories), so 100k memories would still be
   ≈ 6 ms. RAM: 4 KB per memory (40 MB at 10k).
3. **sqlite-vec is not a good default**: 26 ms at 10k (it streams every vector through SQL per
   query) and double the file size. It stays available for "no RAM matrix" deployments.
4. **jieba costs 7 ms** at 10k because `cut_for_search` emits many fragments per query and each
   becomes an OR term; bigram queries are cheaper and better.
5. Build and file size favour SQLite (1.5 s / 49 MB vs 4.7 s / 83 MB).

## The IDF experiment (segment-local statistics)

`bench/bench_idf.py`, 30 035 docs (three memtable flushes in Milvus Lite), 38 probes with a non-empty
oracle answer. Each engine's BM25 top-10 is compared with a **global-statistics oracle that uses that
engine's own tokens, k1, b and IDF formula**, so any gap is the engine's ranking error, not a formula
difference. "Oracle #1 rank" = where the globally best document lands in the engine's list.

| engine (tokens) | top-1 agrees with oracle | oracle #1 at mean rank | top-10 overlap | BM25 query |
|---|---:|---:|---:|---:|
| milvus-lite (jieba) | **29 / 38** | **3.13** | 0.31 | 30.2 ms |
| keepsake FTS5 (jieba) | 38 / 38 | 1.00 | 0.86 | 12.3 ms |
| keepsake FTS5 (bigram) | 38 / 38 | 1.00 | 0.69 | 2.1 ms |

Milvus Lite's segment-local IDF is measurable, not theoretical: once the store spans several
segments, one probe in four gets a different best document than global BM25 would give, and the
globally best memory is pushed to third place on average. FTS5 reproduces global BM25 exactly at
the top (38/38, rank 1.00); its lower top-10 overlap in bigram mode is tie ordering in the tail
(many distractors share the same bigrams and scores), not a ranking error. Insert of 30k rows:
Milvus Lite 3.2 s, FTS5 jieba 2.5 s, FTS5 bigram 1.2 s.

## Verdict for the SQLite trial

The user's rule was: try SQLite, but do not stick to it if it is bad; test whether it works like pure
Milvus and test the speed.

- **Quality**: equal on dense by construction; **better on hybrid** at every size, because the
  lexical leg can be a zh-TW-correct bigram index with global statistics.
- **Speed**: 10–27× faster per query than Milvus Lite in-process, sub-millisecond below a few
  thousand memories, ≈ 2 ms at 10k, linear beyond.
- **Footprint**: one file, no server process, no file-lock daemon, half the disk.

SQLite stays. What would change the verdict later: a store beyond ~10⁵ memories (switch the dense
leg to an ANN index or graduate to Milvus Standalone), or a need for learned-sparse vectors.
