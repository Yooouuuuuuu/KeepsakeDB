# Changelog

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
