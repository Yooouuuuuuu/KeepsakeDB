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
- **Nothing is lost**: every write is in `history`, every search hit with its scores in
  `search_log`, every pipeline decision in `pipeline_log`, all in the same file; an
  operational logger (`configure_logging`) for the rest. `ks.explain(query)` shows the fusion math.
- **One owner, many readers**: the writing process sets `owner=True`; others read the file
  under WAL and send deletes/archives through the built-in command queue.
- **Backups**: `ks.snapshot()` — consistent live copies, text-only option, Anki-style rotation,
  `ks.restore(path)`.
- **CLI**: `keepsake --db her.db stats | list | search "…" --explain | log | delete 42 | backup`.

## Status and stability

`1.0.0` (2026-10-14) — **the API is frozen.** L1 (`Keepsake`, `Memory`), L2 (`Pipeline`, `PipelineConfig`,
`ObserveResult`), L3 (`AgentMemory`), the embedder / LLM protocols, the command queue and the `keepsake` CLI keep
their names and signatures through 1.x; 1.x releases are bug fixes and additive extras only. A change to the file
schema (tables, columns, the meaning of `archived`, `meta` keys that readers rely on) is **2.0**: the file carries
`meta.schema_version` (2 today), a 1.x package refuses a newer file, and a 2.0 package migrates a 1.x file on open.

What 1.0 holds, all tested (`pytest`): hybrid search (dense + BM25, RRF), the CJK bigram tokenizer, two layers
(long facts, short dated notes), trust labels, supersede / archive with reasons, **hold** (kept but not shown, no
hint) and **patch** (tags + meta without re-embedding), turn stamps (`meta.turn_id` / `meta.source`) and
after-the-fact annotation, a fixed subject word rendered as the person's name, logs (history, search, pipeline), the
owner + command queue, backups with rotation, the CLI. Benchmarked against Milvus Lite 3.2.1 with identical bge-m3
vectors on a zh-TW companion corpus: at 10k memories, hybrid recall@5 **0.88 vs 0.78** and **2 ms vs 53 ms** per
query; one file, 49 MB vs 83 MB ([`docs/BENCHMARK.md`](docs/BENCHMARK.md)). Python ≥ 3.11, dependency: numpy.

**Other processes and the file.** One process owns the store (`owner=True`); it keeps the dense index and the FTS in
step with the rows. Anyone else may read the file under WAL at any time. Writes from outside go through the
**command queue** (`INSERT INTO commands` — `delete`, `archive`, `unarchive`, `hold`, `patch`, `backup`, …; the owner
drains it) or, for after-the-fact alignment only, a **meta patch** (`UPDATE memories SET meta = json_patch(meta, …)`:
meta is read from the row each time, never cached). Never write `text`, `tags`, `archived` or `embedding` from
outside: the owner's index would go stale. This is what 556 Production's psyche engine does from the same process.

```
pip install "keepsakedb @ git+https://github.com/Yooouuuuuuu/KeepsakeDB.git@v1.0.0"
pip install -e ".[vec,jieba,pipeline]"      # development, optional extras
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
