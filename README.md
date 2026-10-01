# KeepsakeDB

A lightweight **mixed (hybrid) vector DB for memory**: dense embeddings + BM25 keyword search,
fused by rank, in **one SQLite file** — no server, no Docker, no native extensions.
Built for AI companions and agents that need to remember people, not documents.

> 909 — き・お・く (記憶).

## The one thing to know

KeepsakeDB is layered. **L1 is the store** — a lightweight hybrid vector DB you can use on its
own. **L2 is the pipeline** (extract / tag / merge / archive with an LLM) — **it can be
unplugged**. **L3 is integration** — how an agent wires L1 + L2 into its turn loop.
**The recommended way is all three together; that is how it is designed.** L3 is not tied to
one framework: Hermes, OpenClaw, LangGraph, a hand-written loop — anything that can call
Python before and after a turn.

```
L3  integrate   your agent: context_for(user_text) before the turn, observe(turn) after
L2  pipeline    LLM decides what is worth keeping, tags it, merges conflicts, archives — pluggable
L1  store       SQLite: memories + tags + archive state + history + embeddings + FTS5 BM25
```

## Why hybrid, and why not Milvus Lite

Dense search finds paraphrases (「我的貓」 → the memory about Mochi); BM25 finds the exact
names, numbers and rare words dense models blur (Gogoro, F12, 十八尖山). Milvus made this
pairing standard — `hybrid_search` + `RRFRanker` — and KeepsakeDB copies that model exactly:
two legs, over-fetch, reciprocal-rank fusion with k = 60.

Milvus Lite 3.x (pure Python) also offers it now, but its BM25 statistics are **segment-local**
(an acknowledged TODO in its source), its scoring is a Python loop rebuilt per query, and it
holds a process-level file lock. SQLite's FTS5 keeps **one global index in C**; the dense leg
is exact cosine (FLAT) over numpy or `sqlite-vec`. The full analysis, with sources, is in
[`docs/RESEARCH-milvus.md`](docs/RESEARCH-milvus.md); numbers are in
[`docs/BENCHMARK.md`](docs/BENCHMARK.md). SQLite is a trial, not a dogma: L1 is replaceable.

## L1 in 20 lines

```python
from keepsakedb import Keepsake
from keepsakedb.embedders import OpenAIEmbedder   # any /v1/embeddings endpoint (llama-server, Ollama, …)

ks = Keepsake("her.db", embedder=OpenAIEmbedder("http://127.0.0.1:55602/v1", "bge-m3"))
ks.add("我養了一隻貓叫Mochi，牠最愛吃罐頭", tags=["pets"])
ks.add("十一月我要跑新竹馬拉松，報了半馬", tags=["sports"])

ks.search("我的貓喜歡吃什麼")            # hybrid (dense + BM25, RRF) → [Memory(...), ...]
ks.search("Gogoro", mode="bm25")          # exact keyword, CJK-safe (bigram tokenizer, no deps)

new = ks.supersede(old_id, "馬拉松我升級了，改報全馬", reason="user correction")  # history kept
ks.archive(tag="sports", reason="user finds it painful")   # archive = a state + a reason, not a place
ks.context("最近有什麼運動計畫")
# Long-term memory:
# - [archived: sports — user finds it painful — contents withheld; offer to discuss]
```

- **Tokenizers**: `bigram` (default, zero deps; CJK substring matching down to 2 chars) or
  `jieba`. SQLite's own `unicode61`/`trigram` cannot search Chinese; see the research doc.
- **Dense index**: `numpy` (RAM matrix, exact) or `sqlite-vec` (in the same file).
- **Trust labels** for prompts: `[high]` above a cosine knob (default 0.60), dropped below
  another (0.40) unless BM25 also matched. Calibrate per embedder.
- **Embedder and memory language travel together**: a zh-TW store needs an embedder that
  covers Chinese (bge-m3 does). Switching embedders = `ks.reembed(new)`.

## Status

`0.0.1` — L1 implemented and tested (`pytest`). L2 and L3 are next; the Milvus Lite
comparison is being run. Python ≥ 3.11, dependency: numpy.

```
pip install -e ".[vec,jieba,pipeline]"      # optional extras
pip install -e ".[bench]"                   # milvus-lite, pymilvus, rank-bm25 for the benchmarks
```

## Layout

```
keepsakedb/store/      L1: store.py (Keepsake), tokenize.py, dense.py, fusion.py
keepsakedb/embedders/  OpenAI-compatible HTTP, llama-cpp in-process, cache, test hash
keepsakedb/pipeline/   L2 (pluggable)
keepsakedb/integrate/  L3 guidance + reference adapters
docs/                  RESEARCH-milvus.md, DESIGN.md, BENCHMARK.md
bench/                 corpus + benchmark scripts (KeepsakeDB vs Milvus Lite)
tests/
```

License: MIT.
