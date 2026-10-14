# Changelog

## 1.0.1 — 2026-10-14

- **Bug fix (data safety).** `Pipeline` no longer rewrites the store's subject word just because its config names a
  different one. That migration (the user renamed the subject word) now needs `PipelineConfig(migrate_subject=True)`;
  any other pipeline on the same store — another counterpart, a tool — adopts the store's word instead. A second
  pipeline with `subject="子睿"` used to turn every 「你…」 memory into 「子睿…」 and the main pipeline turned them back
  (taking real 子睿 mentions with it); the 556 cast's stores were damaged this way on 2026-10-14.

## 1.0.0 — 2026-10-14

- **API frozen.** 1.x = bug fixes and additive extras; the file schema is 2 and a breaking change is 2.0 (see README
  "Status and stability" and DESIGN "Migration").
- A 1.x package refuses to open a file whose `meta.schema_version` is newer (it used to open it anyway).
- `keepsakedb.__version__` now matches the package version (it said 0.3.0 at 0.3.1); a test keeps them equal.
- README rewritten short for the public repo; a LICENSE file (MIT) and the SPDX license field in pyproject;
  DESIGN brought up to date: layers, meta keys, hold / patch, the contract for other processes.

## 0.3.1 — 2026-10-14

- **Hold**: `Store.hold(ids | turn_id, on, reason)` (and the `hold` command) marks memories kept but not shown —
  `meta.hold`. They stay embedded and remain observe candidates (no duplicates), but `context_for` and tools that
  honour `Memory.held` skip them silently; unlike `archive`, nothing hints that they exist. The 556 psyche engine
  holds a turn's memories while a hurt from that turn is on its ledger.
- **Patch**: `Store.patch(ids | turn_id, add_tags, meta)` (and the `patch` command) adds tags and merges meta without
  touching the text (no re-embedding). `meta.prefix` is rendered in front of the text by `context_for`
  (e.g. 「（舊帳 2026/10/13）」 on a forgiven hurt that came back as memory).
- `Store.ids_for_turn(turn_id)`: the memories one conversation turn produced (`meta.turn_id`).

## 0.3.0 — 2026-10-07

- `Pipeline.observe(..., turn_id=, source=)` stamps `meta.turn_id` / `meta.source` on every memory the turn
  writes (long facts and short notes).
- `Keepsake.annotate(ids, fields)`, `by_turn(turn_id)`, `annotate_turn(turn_id, fields)`: merge fields such
  as valence, arousal and instance ids into meta after the fact, without re-embedding — an emotion engine
  running in parallel with the observe can align with memory later.

## 0.2.1 — 2026-10-07

- `PipelineConfig.render_subject`: memories keep their stored subject word (556 now stores 你), and the
  injected block renders it as the person's name or address term, so the prompt never confuses 你 with the
  character herself and renaming never rewrites the store.

## 0.2.0 — 2026-10-03

- **Two layers in one file.** `memories.layer` = `long` (tagged facts, trust labels, supersede /
  archive — unchanged) or `short` (dated notes, no tags, a ring capped by `prune_layer`). One owner,
  one backup rotation, one search machinery; `add(..., layer=)`, `search(..., layer=)`,
  `count(layer=)`, `stats()["layers"]`, `prune_layer(layer, keep)`. Existing stores migrate on open.
- **`search_layers()`** — one embedding, every layer searched at once: the bigger layer (in-memory
  count) starts first on its own read-only connection, the smallest runs on the calling thread right
  after; only the final join waits. Results per layer.
- **Pipeline:** the same observe call may return `"note"` (one dated line for the short layer:
  `short_layer`, `short_k`, `short_cap`, `surface`); candidates come from the long layer only.
  `context_for` injects `Facts (long-term)` and `Notes (recent, dated)` from one layered search.

## 0.1.3 — 2026-10-02

- **Removed the recent-exchange layer** (added in 0.1.2). Recording every turn in the store mixed two
  different things: conversation history (verbatim, short-lived, the agent's session store already keeps
  it across restarts) and memory (facts the pipeline judged worth keeping). The store is memory only
  again; `search(..., exclude_tags=)` stays as a generic option.

## 0.1.2 — 2026-10-02

- **Recent-exchange layer** for cross-surface continuity: `add_exchange(surface, q, a)` records one
  turn without an LLM (tags `recent` + `surface:<s>`, 24 h TTL in meta); `recent(limit,
  exclude_surface)`; `sweep_recent()` archives expired ones as `decay` and purges after 7 days
  (run by the owner loop every 300 s). `search(..., exclude_tags=)`.
- Pipeline: `PipelineConfig(surface, recent_k, recent_ttl_hours, self_name)` — `context_for`
  prepends `[recent HH:MM]` exchanges from OTHER surfaces and excludes `recent` from hits and
  from L2 candidates; `observe` records the exchange after the LLM pass.

## 0.1.1 — 2026-10-02

- L2 pipeline retries once (configurable) when the LLM reply is empty or not JSON — found in the
  556 endurance run, where one transient empty reply silently dropped the only mention of a fact.
- Pipeline `consent` op (opens a consent window for an archived tag); reads serialized against
  writes on the shared connection; `context_for` logs candidates with the injected flag.
- Verified in 556: 100 rounds 19/20, exam 20/20 on a 483-round store, 932-round store with
  lossless observes and a direct engine recall of 18/20 (both misses explained).

## 0.1.0 — 2026-10-02

Operations, after the design discussions.

- **Logs, four streams in the same file**: `history` (every write), `search_log` (every hit with
  dense / BM25 / fused score, label, injected or not — on by default), `pipeline_log` (candidates
  shown to the LLM, raw reply, ops applied and skipped, model, latency), and an operational
  logger (`keepsakedb`, JSON lines via `configure_logging`) for open/close, reindex, backups,
  commands, errors and slow queries. `log_retention_days` prunes (default 90).
- **`explain(query)`**: both legs with ranks and scores, the fused list, the FTS5 MATCH string.
- **Ownership and the command queue**: `owner=True` marks the single writer; other processes
  `enqueue` delete / archive / unarchive / backup / restore / reindex / reembed and the owner
  drains before every search and on `start_owner_loop`. Heartbeat (`owner_alive`). Every
  connection checks `PRAGMA data_version` and reloads rows changed elsewhere, so a direct
  write from another process is visible within one query. `busy_timeout` on every connection.
- **Backups**: `backup(dest, include_embeddings)` (online backup API, atomic rename),
  `snapshot()` with Anki-style rotation (7 daily / 4 weekly / 12 monthly), `restore(path)` in
  place (text-only snapshots re-embed), daily auto-snapshot and one before risky operations
  when `backup_dir` is set.
- **`keepsake` CLI**: stats, list, show, search (`--explain`), tags, log, plog, delete / archive /
  unarchive, backup, restore, commands. Writes go through the owner when one is alive.
- Weighted fusion defaults to min-max; a superseded memory is its own state (never surfaced).
- Benchmarks: fusion sweep, reranking (decided against), Milvus Lite comparison.

## 0.0.1 — 2026-10-01

L1 store (SQLite + FTS5 BM25 + numpy / sqlite-vec dense, RRF), L2 pipeline, L3 integration,
Milvus research, benchmark harness.
