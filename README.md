# KeepsakeDB

A memory store for AI companions and agents: **dense embeddings + BM25 keyword search, fused by
rank, in one SQLite file.** No server, no Docker, no native extensions. Built to remember people,
not documents.

```python
from keepsakedb import Keepsake
from keepsakedb.embedders import OpenAIEmbedder   # any /v1/embeddings endpoint (llama-server, Ollama, …)

ks = Keepsake("her.db", embedder=OpenAIEmbedder("http://127.0.0.1:55602/v1", "bge-m3"))
ks.add("我養了一隻貓叫Mochi，牠最愛吃罐頭", tags=["pets"])
ks.add("十一月我要跑新竹馬拉松，報了半馬", tags=["sports"])

ks.search("我的貓喜歡吃什麼")                      # hybrid: dense finds the paraphrase, BM25 the exact name
ks.supersede(old_id, "馬拉松我改報全馬", reason="user correction")   # history kept
ks.archive(tag="sports", reason="user finds it painful")            # a state with a reason, not a place
ks.context("最近有什麼運動計畫")                    # the block you inject into the prompt
```

## Install

```
pip install "keepsakedb @ git+https://github.com/Yooouuuuuuu/KeepsakeDB.git@v1.0.0"
pip install -e ".[vec,jieba,pipeline]"      # development; extras: sqlite-vec index, jieba tokenizer, LLM pipeline
```

Python ≥ 3.11. One dependency: numpy.

## What it does

- **Hybrid search**: cosine over embeddings (numpy or `sqlite-vec`) + FTS5 BM25, reciprocal-rank
  fusion. Chinese works out of the box: a bigram tokenizer, no dependencies (`jieba` optional).
- **Two layers**: long-term facts with tags and trust labels; short-term dated notes in a ring.
- **Nothing is overwritten blindly**: `supersede` keeps history, `archive` keeps the text and
  tells the prompt only that a topic is withheld, `hold` hides a memory without a trace until it
  is released, `patch` changes tags and metadata without re-embedding.
- **An LLM pipeline you can unplug** (L2): after each turn it decides what is worth keeping,
  tags it, merges conflicts, archives on request. The store (L1) works alone.
- **Agent integration** (L3): `context_for(user_text)` before the turn, `observe(turn)` after.
  Hermes, OpenClaw, LangGraph or a plain loop.
- **One owner, many readers**: one process writes and keeps the index; others read under WAL and
  send writes through a command queue in the same file.
- **Logs and backups**: every write, search hit and pipeline decision is in the file;
  `ks.explain(query)` shows the fusion; snapshots with rotation and restore.
- **CLI**: `keepsake --db her.db stats | list | search "…" --explain | delete 42 | backup`.

Against Milvus Lite 3.2.1 with identical bge-m3 vectors on a zh-TW companion corpus, at 10k
memories: hybrid recall@5 0.88 vs 0.78, 2 ms vs 53 ms per query, one 49 MB file vs 83 MB
([`docs/BENCHMARK.md`](docs/BENCHMARK.md), [`docs/RESEARCH-milvus.md`](docs/RESEARCH-milvus.md)).

## Stability

**1.0.0 — the API is frozen.** 1.x releases are bug fixes and additive extras. The file carries
`meta.schema_version`; a breaking schema change is 2.0, which migrates 1.x files on open, and a
1.x package refuses a newer file. Other processes read freely and write only through the command
queue or a metadata patch; see [`docs/DESIGN.md`](docs/DESIGN.md).

## Layout

```
keepsakedb/store/      L1: the store (search, layers, archive / hold / patch, logs, owner + queue, backups)
keepsakedb/pipeline/   L2: the LLM pipeline (pluggable)
keepsakedb/integrate/  L3: wiring into an agent loop
keepsakedb/embedders/  OpenAI-compatible HTTP, llama-cpp in-process, cache, test hash
docs/                  DESIGN.md, BENCHMARK.md, RESEARCH-milvus.md
tests/                 pytest
```

## License

[MIT](LICENSE).
