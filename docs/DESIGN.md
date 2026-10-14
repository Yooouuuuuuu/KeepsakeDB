# KeepsakeDB design

## Layers

```
L3  integrate   keepsakedb.integrate.AgentMemory — before_turn / after_turn / end_session
L2  pipeline    keepsakedb.pipeline.Pipeline     — observe (LLM decides add/update/supersede/archive), context_for, consent windows
L1  store       keepsakedb.store.Keepsake        — SQLite file: memories, tags, archive state, history, embeddings, FTS5 BM25, sqlite-vec (opt.)
```

L1 depends on numpy only. L2 depends on L1 and a chat LLM. L3 is a 40-line convenience; copy
it if your loop is different. **Recommended: L1 + L2 through L3** — the pipeline is what turns
a vector DB into a memory. **L1 alone** is a valid product: a hybrid store with archive and
history semantics, for people who run their own extraction.

## L1 — the store

### Data model

| Table | Holds |
|---|---|
| `memories` | `id`, `text`, `tokens` (pre-tokenized), `tags` (JSON), `meta` (JSON), `archived`, `archive_reason`, `created_at`, `updated_at`, `embedding` (float32 blob), `embedder` |
| `memory_tags` | normalized `(memory_id, tag)` for fast tag queries |
| `history` | append-only: `add / update / supersede / archive / unarchive / hold / release / patch / delete` with old/new text and reason |
| `memories_fts` | FTS5 (`unicode61`) over `tokens` — the BM25 inverted index |
| `memories_vec` | optional sqlite-vec `vec0(id, embedding float[d] cosine, flag)` |
| `meta` | `schema_version` (2), tokenizer name, dims, the subject word, the owner heartbeat, last snapshot |

Each memory has a `layer` (`long` facts with tags; `short` dated notes in a ring) and a JSON `meta` whose keys the
callers own: `turn_id` / `source` (stamped by the observe), `valence` / `arousal` / `instance_ids` (annotated later by
an emotion engine), `hold` (kept but not shown: `context_for` and tools that honour `Memory.held` skip it silently —
unlike archive, no hint), `prefix` (rendered in front of the text, e.g. 「（舊帳 2026/10/13）」), `old_account_date`.

One file per companion = Milvus's `user_id`/partition isolation without a server.

### Tokenization (the CJK problem)

FTS5's `unicode61` keeps a CJK run as one token; `trigram` needs ≥ 3 characters. Both miss
two-character Chinese words. `store/tokenize.py` pre-tokenizes:

- `bigram` (default): CJK runs → overlapping character bigrams, Latin → lowercased words.
  Query = bag of bigrams OR-ed (BM25 sums them, like Milvus). `phrase=True` turns a run into
  an FTS5 phrase = exact substring lookup (names, keywords).
- `jieba`: word segmentation (`cut_for_search`, as Milvus Lite). Exact words only.

The tokenizer is recorded in `meta`; `reindex()` switches it.

### Search

```
dense leg  : cosine top-(k·oversample) over normalized vectors (numpy matrix or sqlite-vec)
bm25 leg   : FTS5 bm25() top-(k·oversample), global statistics, k1 = 1.2, b = 0.75
fusion     : RRF  score = Σ 1/(60 + rank)        (Milvus RRFRanker default; no leg weights — by rank)
             or weighted Σ wᵢ·normᵢ(score)        (min-max default — won the sweep; arctan like Standalone)
filters    : archived state (exclude | include | meta), tag
labels     : [high] when cosine ≥ high (0.60); dropped when < drop (0.40) unless BM25 matched
logging    : log_searches=True records every hit's dense/BM25/fused score + whether it was injected
             (search_log table; score_percentiles() for calibrating high/drop on real traffic)
```

Defaults are benchmark-backed (`docs/BENCHMARK.md`): RRF beats every dense-heavy weighting
once the store has thousands of memories; the document-RAG habit of 0.7–0.8 dense does not
transfer to one-line facts. `high`/`drop` are the only quality knobs still uncalibrated — turn on
`log_searches` for the first real run and set them from the distribution.

`context(query)` renders the injection block. Archived hits appear as meta-info only:
`[archived: sports — user finds it painful — contents withheld; offer to discuss]`.

### Archive = a state with a reason, not a place

`archive(ids | tag, reason)` keeps the memory embedded and searchable, hides it from normal
search, records the reason, logs history. `unarchive` reverses it. `supersede(old, text,
reason)` is the correction primitive: the new memory inherits the tags and points at the old
one (`meta.supersedes`), the old one is archived with the reason. `delete` is the only hard
operation (user-commanded purge); history keeps the text.

### Knobs

`Keepsake(path, embedder, tokenizer="bigram", dense_index="numpy", rrf_k=60, high=0.60,
drop=0.40, oversample=3)`. Thresholds are per-embedder and should be calibrated on real score
distributions (556's bge-m3 data: correct paraphrase hits 0.46–0.65, noise ≈ 0.39).

### Logs (nothing is lost; negligible when on, zero when off)

| stream | where | what | cost |
|---|---|---|---|
| `history` | table | every add / update / supersede / archive / unarchive / delete with old & new text and reason | one insert inside the write's own transaction |
| `search_log` | table, `log_searches` (default on) | every hit of every search: rank, fused, dense, BM25, label, injected-or-not for `context()` | ≈ tens of µs per query under WAL, no extra fsync |
| `pipeline_log` | table, `log_pipeline` (default on) | per observe: candidate ids shown, raw LLM reply, ops applied / skipped, error, model, ms (not the prompt) | one insert per turn |
| operational | `logging.getLogger("keepsakedb")` | open/close, reindex, reembed, backups, commands, errors, **slow queries** over `slow_ms` | nothing unless a handler is attached (`configure_logging(path)` → JSON lines) |

`log_retention_days` (90) prunes the two log tables and finished commands at open.
Host caveat: a host may silence Python logging globally (Hermes one-shot `-z` mode calls
`logging.disable`), which mutes the operational stream only; the three SQLite streams are
unaffected, which is why the facts live in the file and not in the log.
`explain(query)` returns the fusion math for one query (never logged). `score_percentiles()`
turns `search_log` into the calibration data for `high` / `drop`.

### Ownership, the command queue, and other processes

One process writes (`owner=True`): it holds the dense cache in RAM, drains the `commands`
table before every search and on a timer (`start_owner_loop`), and publishes a heartbeat.
Any other process (556d's memory page, the CLI) reads freely under WAL and sends writes as
commands: `enqueue("delete", {"ids": [...]})` → the owner executes it → `wait_command` sees
the result. A queued command also survives the owner being offline: it runs at the next
drain. Supported: delete, archive, unarchive, backup, restore, reindex, reembed, ping.

Safety net: every connection checks `PRAGMA data_version` (changes only when *another*
connection commits) before a search and reloads the rows named in `history` since its last
look, so even a direct write from elsewhere is visible within one query. `busy_timeout`
(5 s) on every connection means SQLite-level contention never surfaces as an error.

### Backups

`backup(dest, include_embeddings=True)` uses SQLite's online backup API (consistent while
live) and renames a temp file into place, so a synced folder only ever sees complete files.
Text-only snapshots drop the vectors (≈ 80 % of the file); restoring one re-embeds.
`snapshot()` writes `<name>-YYYYMMDD-HHMMSS[-text].db` into `backup_dir` and rotates
Anki-style: newest per day for 7 days, per week for 4 weeks, per month for 12 months.
With `backup_dir` set, a snapshot is taken on the first write of each day and before
reindex / reembed / purge-by-tag / restore. `restore(path)` copies a snapshot into the open
database in place (readers never see a torn file). Never copy a live `.db` by hand (its
`-wal` file holds recent commits); keep the file on a local disk (WAL is unsafe on network
shares). Whether `backup_dir` points at a synced cloud folder is the user's choice.

### `keepsake` CLI

`keepsake --db her.db stats | list | show ID | search Q [--explain] | tags | log | plog |
delete | archive | unarchive | backup | restore | commands` (`--json` for machines,
`--embedder URL` for the dense leg). Write commands go through the owner when one is alive.

## L2 — the pipeline

`Pipeline(store, llm, PipelineConfig(language="zh-TW", max_candidates=8, consent_turns=100,
context_k=5, prefilter=None))`

`observe(user_text, assistant_text=None)`:

1. optional `prefilter` (regex/length) can skip the LLM for obvious small talk;
2. candidates = hybrid top-8 over active **and archived** memories + the tag groups;
3. one LLM call → JSON `ops`: `add {text, tags}`, `update {id, text, reason}`,
   `supersede {id, text, reason}`, `archive {tag | id, reason}`, `unarchive {tag | id}`, `noop`;
4. ops are validated (ids must be candidates, text non-empty, tags slugged) and applied; an
   LLM/JSON failure returns `ObserveResult(error=…)` and never breaks the turn.

What the prompt enforces (learned from the 556 1000-round test): user facts only (no assistant
advice leakage), memories written in the configured language (no mixed-language duplicates),
restatements are `noop`, contradictions are `supersede` with recency + explicit correction
beating repetition, tags are lowercase English slugs reused across the store.

`context_for(user_text)` = the injection block; archived groups are meta-info unless the
user granted consent: `grant_consent(tag, turns)` opens a window of N turns (`turn()` counts),
`end_session()` closes every window. Permanent recovery is `store.unarchive`.

## L3 — integration (any agent)

Two hooks and a session boundary. `AgentMemory(store, llm, background=True)`:

| Hook | Call | Notes |
|---|---|---|
| before the model runs | `block = mem.before_turn(user_text)` | inject `block` into the system/context prompt |
| after the reply exists | `mem.after_turn(user_text, reply)` | observe; background thread by default so the turn never waits for the LLM |
| session ends | `mem.end_session()` | consent windows close |

Framework mapping:

- **Hermes Agent** `MemoryProvider`: `prefetch(query)` → `before_turn`; `sync_turn(user, assistant)`
  → `after_turn`; `on_session_end` → `end_session`; `on_pre_compress` → optionally `observe`
  the turns about to be summarized. (556 ships this as `plugins/kokoro`.)
- **OpenClaw / plugin systems**: the pre-prompt and post-reply hooks.
- **LangGraph**: a node before the model node (adds the block to state) and one after.
- **Plain loop**: three lines around your `chat()` call.

The LLM for L2 can differ from the chat model (556 uses a fast flash model); the embedder
must cover the memory language.

## Not in scope (deliberately)

- A reranker in the per-turn path — measured (`docs/BENCHMARK.md`): a cross-encoder reorders
  hybrid's top-5 (+R@1) without improving what gets injected (R@5), for ≈ 0.5 s of CPU per turn;
  it cannot rescue dense-only or BM25-only retrieval. LLM judgement over candidates lives in L2's
  background observe pass instead.
- Curated "head" memory (always-injected top facts) — that is the agent's own file (Hermes
  `MEMORY.md`); L1 is the long tail.
- Verbatim transcripts — the agent's session store already has them (Hermes FTS5 sessions).
- Learned-sparse vectors (bge-m3 lexical weights) — a possible second lexical leg later.
- Multi-process writers; ≥ 10⁶ memories (graduate to Milvus Standalone, same model).

## Migration

**Versions.** 1.x reads schema ≤ 2 and migrates additive columns on open (`layer` was one); a file with a
newer `schema_version` is refused. A breaking schema change is 2.0, which migrates a 1.x file on open.

From mem0/Qdrant: read payloads, `add_many` with fresh embeddings — minutes at companion
scale. Switching embedders: `reembed(new_embedder)`. Switching tokenizers: `reindex("jieba")`.
