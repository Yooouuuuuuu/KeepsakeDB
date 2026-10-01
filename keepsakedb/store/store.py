"""L1 — the store. One SQLite file holding memories, tags, archive state, history,
embeddings, an FTS5 BM25 index, (optionally) a sqlite-vec index, the logs and the
command queue.

Hybrid search = dense cosine leg + BM25 leg -> RRF (or weighted) fusion, the same
shape as Milvus ``hybrid_search`` with ``RRFRanker``. Unlike Milvus Lite, BM25
statistics are global: FTS5 keeps one index for the whole table, so IDF/avgdl never
depend on which flush segment a memory landed in.

Ownership: exactly one process should write (``owner=True``) because it holds the
dense cache in RAM. Other processes read freely (WAL) and send writes through the
``commands`` table; the owner drains it before every search and on a timer. As a safety
net, every connection checks ``PRAGMA data_version`` before searching and reloads the
rows another connection changed (bounded staleness: one query).
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from . import fusion
from .dense import ACTIVE, ARCHIVED, NumpyDense, SqliteVecDense
from .tokenize import Tokenizer, fts_query, get_tokenizer, index_string

log = logging.getLogger("keepsakedb")

SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS memories(
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    tokens TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]',
    meta TEXT NOT NULL DEFAULT '{}',
    archived INTEGER NOT NULL DEFAULT 0,
    archive_reason TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    embedding BLOB,
    embedder TEXT
);
CREATE TABLE IF NOT EXISTS memory_tags(memory_id INTEGER NOT NULL, tag TEXT NOT NULL,
    PRIMARY KEY(memory_id, tag));
CREATE INDEX IF NOT EXISTS idx_tag ON memory_tags(tag);
CREATE INDEX IF NOT EXISTS idx_archived ON memories(archived);
CREATE TABLE IF NOT EXISTS history(
    id INTEGER PRIMARY KEY, memory_id INTEGER NOT NULL, op TEXT NOT NULL,
    old_text TEXT, new_text TEXT, reason TEXT, at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_history_mem ON history(memory_id);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(tokens, tokenize='unicode61');
CREATE TABLE IF NOT EXISTS search_log(
    id INTEGER PRIMARY KEY, at REAL NOT NULL, query TEXT NOT NULL, mode TEXT NOT NULL,
    memory_id INTEGER NOT NULL, rank INTEGER NOT NULL, fused REAL, dense REAL, bm25 REAL,
    label TEXT, injected INTEGER);
CREATE INDEX IF NOT EXISTS idx_search_log_at ON search_log(at);
CREATE TABLE IF NOT EXISTS pipeline_log(
    id INTEGER PRIMARY KEY, at REAL NOT NULL, user_text TEXT NOT NULL, candidates TEXT,
    raw TEXT, applied TEXT, skipped TEXT, error TEXT, model TEXT, ms REAL);
CREATE INDEX IF NOT EXISTS idx_pipeline_log_at ON pipeline_log(at);
CREATE TABLE IF NOT EXISTS commands(
    id INTEGER PRIMARY KEY, op TEXT NOT NULL, payload TEXT NOT NULL DEFAULT '{}',
    requested_at REAL NOT NULL, requested_by TEXT, done_at REAL, result TEXT);
"""

_COLS = "id, text, tags, meta, archived, archive_reason, created_at, updated_at"
_LOG_COLS = ("id", "at", "query", "mode", "memory_id", "rank", "fused", "dense", "bm25", "label", "injected")
_PLOG_COLS = ("id", "at", "user_text", "candidates", "raw", "applied", "skipped", "error", "model", "ms")
_CMD_COLS = ("id", "op", "payload", "requested_at", "requested_by", "done_at", "result")
_SNAP_RE = re.compile(r"^(?P<stem>.+)-(?P<ts>\d{8}-\d{6})(?P<text>-text)?\.db$")
COMMAND_OPS = ("delete", "archive", "unarchive", "backup", "restore", "reindex", "reembed", "ping")


@dataclass
class Memory:
    id: int
    text: str
    tags: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    archived: bool = False          # hidden from normal search (topic suppression OR superseded)
    superseded: bool = False        # replaced by a correction: never shown, not even as meta-info
    archive_reason: Optional[str] = None
    created_at: float = 0.0
    updated_at: float = 0.0
    # search-time fields
    score: float = 0.0             # fused score
    dense: Optional[float] = None  # cosine, if the dense leg saw it
    bm25: Optional[float] = None   # positive BM25, if the BM25 leg saw it
    label: str = "plain"           # high | plain | archived

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class Keepsake:
    def __init__(self, path: str, embedder=None, tokenizer: "str | Tokenizer" = "bigram",
                 dense_index: str = "numpy", dims: Optional[int] = None,
                 rrf_k: float = 60.0, high: float = 0.60, drop: float = 0.40,
                 oversample: int = 3, owner: bool = False, backup_dir: Optional[str] = None,
                 snapshot_text_only: bool = False, log_searches: bool = True,
                 log_pipeline: bool = True, log_retention_days: Optional[int] = 90,
                 slow_ms: float = 50.0, busy_timeout: float = 5.0) -> None:
        """owner=True marks this process as the single writer (drains the command queue,
        publishes a heartbeat). backup_dir enables a daily snapshot on the first write of
        each day plus one before risky operations (reindex / reembed / purge / restore).
        log_searches / log_pipeline record every hit and every pipeline decision in the
        same file (``search_log``, ``pipeline_log``); log_retention_days prunes them."""
        self.path = path
        self.embedder = embedder
        self.tok = get_tokenizer(tokenizer)
        self.rrf_k = rrf_k
        self.high = high
        self.drop = drop
        self.oversample = oversample
        self.owner = owner
        self.backup_dir = backup_dir
        self.snapshot_text_only = snapshot_text_only
        self.log_searches = log_searches
        self.log_pipeline = log_pipeline
        self.log_retention_days = log_retention_days
        self.slow_ms = slow_ms
        self._lock = threading.RLock()
        self._owner_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._last_heartbeat = 0.0
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=busy_timeout)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(_SCHEMA)
        self._set_meta_default("schema_version", str(SCHEMA_VERSION))
        self._set_meta_default("tokenizer", self.tok.name)
        if self._meta("tokenizer") != self.tok.name:
            raise ValueError(f"store was built with tokenizer {self._meta('tokenizer')!r}; call reindex() to switch")
        stored_dims = self._meta("dims")
        self.dims = dims or (int(stored_dims) if stored_dims else None) or getattr(embedder, "dims", None)
        self._dense_kind = dense_index
        self.dense = None
        if self.dims:
            self._init_dense()
        self._data_version = self._pragma_data_version()
        self._last_history_id = self._max_history_id()
        if log_retention_days:
            self.prune_logs(log_retention_days)
        log.info("open %s: %d memories, dims=%s, tokenizer=%s, owner=%s", path,
                 self.count(include_archived=True), self.dims, self.tok.name, owner)

    # ---------------------------------------------------------------- meta
    def _meta(self, key: str) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))

    def _set_meta_default(self, key: str, value: str) -> None:
        self.conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)", (key, value))

    def _pragma_data_version(self) -> int:
        return int(self.conn.execute("PRAGMA data_version").fetchone()[0])

    def _max_history_id(self) -> int:
        return int(self.conn.execute("SELECT COALESCE(MAX(id), 0) FROM history").fetchone()[0])

    def _init_dense(self) -> None:
        self._set_meta("dims", str(self.dims))
        if self._dense_kind == "numpy":
            self.dense = NumpyDense(self.dims)
            rows = self.conn.execute("SELECT id, embedding, archived FROM memories WHERE embedding IS NOT NULL")
            self.dense.load((i, b, ARCHIVED if a else ACTIVE) for i, b, a in rows)
        elif self._dense_kind == "sqlite-vec":
            self.dense = SqliteVecDense(self.conn, self.dims)
        else:
            raise ValueError("dense_index must be 'numpy' or 'sqlite-vec'")

    # ------------------------------------------------------------- embedding
    def _embed(self, texts: Sequence[str]) -> Optional[np.ndarray]:
        if self.embedder is None or not texts:
            return None
        vecs = self.embedder.embed(texts)
        if self.dims is None:
            self.dims = int(vecs.shape[1])
            self._init_dense()
        elif vecs.shape[1] != self.dims:
            raise ValueError(f"embedder returned {vecs.shape[1]} dims, store has {self.dims}")
        return vecs

    # ----------------------------------------------------------------- write
    def add(self, text: str, tags: Iterable[str] = (), meta: Optional[Dict[str, Any]] = None,
            embedding: Optional[np.ndarray] = None, ts: Optional[float] = None) -> int:
        emb = None if embedding is None else np.asarray([embedding], dtype=np.float32)
        return self.add_many([(text, list(tags), meta or {})], embeddings=emb, ts=ts)[0]

    def add_many(self, items: Sequence[Tuple[str, List[str], Dict[str, Any]]],
                 embeddings: Optional[np.ndarray] = None, ts: Optional[float] = None) -> List[int]:
        if not items:
            return []
        self._auto_snapshot()
        now = ts or time.time()
        if embeddings is None:
            embeddings = self._embed([t for t, _, _ in items])
        elif self.dims is None:
            self.dims = int(embeddings.shape[1])
            self._init_dense()
        ids: List[int] = []
        ename = getattr(self.embedder, "name", None)
        with self._lock:
            self.conn.execute("BEGIN")
            try:
                for i, (text, tags, meta) in enumerate(items):
                    tags = sorted(set(tags))
                    toks = index_string(self.tok, text)
                    blob = embeddings[i].astype(np.float32).tobytes() if embeddings is not None else None
                    cur = self.conn.execute(
                        "INSERT INTO memories(text, tokens, tags, meta, created_at, updated_at, embedding, embedder)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (text, toks, json.dumps(tags, ensure_ascii=False), json.dumps(meta, ensure_ascii=False),
                         now, now, blob, ename))
                    mid = int(cur.lastrowid)
                    self.conn.execute("INSERT INTO memories_fts(rowid, tokens) VALUES (?, ?)", (mid, toks))
                    self.conn.executemany("INSERT INTO memory_tags(memory_id, tag) VALUES (?, ?)",
                                          [(mid, t) for t in tags])
                    self.conn.execute("INSERT INTO history(memory_id, op, new_text, at) VALUES (?, 'add', ?, ?)",
                                      (mid, text, now))
                    if embeddings is not None and self.dense is not None:
                        self.dense.add(mid, embeddings[i], ACTIVE)
                    ids.append(mid)
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
            self._last_history_id = self._max_history_id()
        return ids

    def update(self, mid: int, text: Optional[str] = None, tags: Optional[Iterable[str]] = None,
               meta: Optional[Dict[str, Any]] = None, reason: Optional[str] = None,
               embedding: Optional[np.ndarray] = None) -> None:
        cur = self.get(mid)
        if cur is None:
            raise KeyError(mid)
        now = time.time()
        with self._lock:
            self.conn.execute("BEGIN")
            try:
                if text is not None and text != cur.text:
                    toks = index_string(self.tok, text)
                    vec = embedding
                    if vec is None:
                        e = self._embed([text])
                        vec = e[0] if e is not None else None
                    self.conn.execute(
                        "UPDATE memories SET text=?, tokens=?, updated_at=?, embedding=?, embedder=? WHERE id=?",
                        (text, toks, now, vec.astype(np.float32).tobytes() if vec is not None else None,
                         getattr(self.embedder, "name", None), mid))
                    self.conn.execute("DELETE FROM memories_fts WHERE rowid=?", (mid,))
                    self.conn.execute("INSERT INTO memories_fts(rowid, tokens) VALUES (?, ?)", (mid, toks))
                    if vec is not None and self.dense is not None:
                        self.dense.add(mid, vec, ARCHIVED if cur.archived else ACTIVE)
                    self.conn.execute(
                        "INSERT INTO history(memory_id, op, old_text, new_text, reason, at) VALUES (?, 'update', ?, ?, ?, ?)",
                        (mid, cur.text, text, reason, now))
                if tags is not None:
                    tl = sorted(set(tags))
                    self.conn.execute("UPDATE memories SET tags=?, updated_at=? WHERE id=?",
                                      (json.dumps(tl, ensure_ascii=False), now, mid))
                    self.conn.execute("DELETE FROM memory_tags WHERE memory_id=?", (mid,))
                    self.conn.executemany("INSERT INTO memory_tags(memory_id, tag) VALUES (?, ?)", [(mid, t) for t in tl])
                if meta is not None:
                    self.conn.execute("UPDATE memories SET meta=?, updated_at=? WHERE id=?",
                                      (json.dumps(meta, ensure_ascii=False), now, mid))
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
            self._last_history_id = self._max_history_id()

    def supersede(self, old_id: int, text: str, reason: str = "superseded",
                  tags: Optional[Iterable[str]] = None, embedding: Optional[np.ndarray] = None) -> int:
        """Record a correction: the old memory is archived (with the reason) and a new one
        is added carrying the old tags unless overridden. History links both."""
        old = self.get(old_id)
        if old is None:
            raise KeyError(old_id)
        new_id = self.add(text, tags if tags is not None else old.tags, dict(old.meta, supersedes=old_id),
                          embedding=embedding)
        now = time.time()
        with self._lock, self.conn:
            # state 2 = superseded: hidden like archived, but never surfaced as meta-info or by consent
            self.conn.execute("UPDATE memories SET archived=2, archive_reason=?, updated_at=? WHERE id=?",
                              (reason, now, old_id))
            self.conn.execute(
                "INSERT INTO history(memory_id, op, old_text, new_text, reason, at) VALUES (?, 'supersede', ?, ?, ?, ?)",
                (old_id, old.text, text, f"-> {new_id}: {reason}", now))
            if self.dense is not None:
                self.dense.set_flag(old_id, ARCHIVED)
        self._last_history_id = self._max_history_id()
        return new_id

    def _ids_for(self, ids: Optional[Iterable[int]], tag: Optional[str], archived: Optional[int] = None) -> List[int]:
        out: Set[int] = set(ids or [])
        if tag is not None:
            out.update(r[0] for r in self.conn.execute("SELECT memory_id FROM memory_tags WHERE tag=?", (tag,)))
        if archived is not None and out:  # only memories currently in the given state
            q = f"SELECT id FROM memories WHERE archived=? AND id IN ({','.join('?' * len(out))})"
            out = {r[0] for r in self.conn.execute(q, [archived, *sorted(out)])}
        return sorted(out)

    def archive(self, ids: Optional[Iterable[int]] = None, tag: Optional[str] = None, reason: str = "") -> int:
        """Archive = a state plus a reason, not a place: memories stay embedded and
        searchable, but ``search`` hides them and ``context`` reports them as meta-info only.
        Already-archived memories are left untouched (their reason and history stay)."""
        targets = self._ids_for(ids, tag, archived=0)
        now = time.time()
        with self._lock, self.conn:
            for mid in targets:
                self.conn.execute("UPDATE memories SET archived=1, archive_reason=?, updated_at=? WHERE id=?",
                                  (reason, now, mid))
                self.conn.execute("INSERT INTO history(memory_id, op, reason, at) VALUES (?, 'archive', ?, ?)",
                                  (mid, reason, now))
                if self.dense is not None:
                    self.dense.set_flag(mid, ARCHIVED)
        self._last_history_id = self._max_history_id()
        return len(targets)

    def unarchive(self, ids: Optional[Iterable[int]] = None, tag: Optional[str] = None, reason: str = "") -> int:
        targets = self._ids_for(ids, tag, archived=1)
        now = time.time()
        with self._lock, self.conn:
            for mid in targets:
                self.conn.execute("UPDATE memories SET archived=0, archive_reason=NULL, updated_at=? WHERE id=?",
                                  (now, mid))
                self.conn.execute("INSERT INTO history(memory_id, op, reason, at) VALUES (?, 'unarchive', ?, ?)",
                                  (mid, reason, now))
                if self.dense is not None:
                    self.dense.set_flag(mid, ACTIVE)
        self._last_history_id = self._max_history_id()
        return len(targets)

    def delete(self, ids: Optional[Iterable[int]] = None, tag: Optional[str] = None, reason: str = "") -> int:
        """Hard delete (user-commanded purge). History keeps the text. A purge by tag takes
        a snapshot first when backup_dir is set."""
        targets = self._ids_for(ids, tag)
        if tag is not None and targets:
            self._auto_snapshot(force=True, reason=f"purge tag {tag}")
        now = time.time()
        n = 0
        with self._lock, self.conn:
            for mid in targets:
                row = self.conn.execute("SELECT text FROM memories WHERE id=?", (mid,)).fetchone()
                if row is None:
                    continue
                self.conn.execute(
                    "INSERT INTO history(memory_id, op, old_text, reason, at) VALUES (?, 'delete', ?, ?, ?)",
                    (mid, row[0], reason, now))
                self.conn.execute("DELETE FROM memories WHERE id=?", (mid,))
                self.conn.execute("DELETE FROM memories_fts WHERE rowid=?", (mid,))
                self.conn.execute("DELETE FROM memory_tags WHERE memory_id=?", (mid,))
                if self.dense is not None:
                    self.dense.remove(mid)
                n += 1
        self._last_history_id = self._max_history_id()
        return n

    # ------------------------------------------------------------------ read
    @staticmethod
    def _row(r: Tuple) -> Memory:
        return Memory(id=r[0], text=r[1], tags=json.loads(r[2]), meta=json.loads(r[3]), archived=bool(r[4]),
                      superseded=(r[4] == 2), archive_reason=r[5], created_at=r[6], updated_at=r[7])

    def get(self, mid: int) -> Optional[Memory]:
        r = self.conn.execute(f"SELECT {_COLS} FROM memories WHERE id=?", (mid,)).fetchone()
        return self._row(r) if r else None

    def get_many(self, ids: Sequence[int]) -> Dict[int, Memory]:
        if not ids:
            return {}
        q = f"SELECT {_COLS} FROM memories WHERE id IN ({','.join('?' * len(ids))})"
        return {r[0]: self._row(r) for r in self.conn.execute(q, list(ids))}

    def all(self, include_archived: bool = True, limit: Optional[int] = None) -> List[Memory]:
        cond = "" if include_archived else " WHERE archived=0"
        lim = f" LIMIT {int(limit)}" if limit else ""
        return [self._row(r) for r in self.conn.execute(f"SELECT {_COLS} FROM memories{cond} ORDER BY id{lim}")]

    def by_tag(self, tag: str, include_archived: bool = False) -> List[Memory]:
        cond = "" if include_archived else " AND m.archived=0"
        cols = ", ".join("m." + c for c in _COLS.split(", "))
        return [self._row(r) for r in self.conn.execute(
            f"SELECT {cols} FROM memories m JOIN memory_tags t ON t.memory_id=m.id WHERE t.tag=?{cond} ORDER BY m.id",
            (tag,))]

    def tags(self, include_archived: bool = False) -> Dict[str, int]:
        cond = "" if include_archived else " WHERE m.archived=0"
        return dict(self.conn.execute(
            f"SELECT t.tag, count(*) FROM memory_tags t JOIN memories m ON m.id=t.memory_id{cond} "
            "GROUP BY t.tag ORDER BY 2 DESC, 1"))

    def history(self, mid: int) -> List[Dict[str, Any]]:
        cols = ("id", "memory_id", "op", "old_text", "new_text", "reason", "at")
        return [dict(zip(cols, r)) for r in self.conn.execute(
            "SELECT id, memory_id, op, old_text, new_text, reason, at FROM history WHERE memory_id=? ORDER BY id",
            (mid,))]

    def count(self, include_archived: bool = False) -> int:
        cond = "" if include_archived else " WHERE archived=0"
        return int(self.conn.execute(f"SELECT count(*) FROM memories{cond}").fetchone()[0])

    def stats(self) -> Dict[str, Any]:
        c = self.conn
        one = lambda q: c.execute(q).fetchone()[0]  # noqa: E731
        owner = self.owner_info()
        return {
            "path": self.path, "memories": self.count(), "archived": one("SELECT count(*) FROM memories WHERE archived=1"),
            "superseded": one("SELECT count(*) FROM memories WHERE archived=2"), "dims": self.dims,
            "tokenizer": self.tok.name, "embedder": self._meta("last_embedder") or getattr(self.embedder, "name", None),
            "tags": len(self.tags(include_archived=True)), "history_rows": one("SELECT count(*) FROM history"),
            "search_log_rows": one("SELECT count(*) FROM search_log"),
            "pipeline_log_rows": one("SELECT count(*) FROM pipeline_log"),
            "pending_commands": one("SELECT count(*) FROM commands WHERE done_at IS NULL"),
            "owner": owner, "owner_alive": self.owner_alive(),
            "last_snapshot": self._meta("last_snapshot"), "file_bytes": os.path.getsize(self.path) if os.path.exists(self.path) else 0,
        }

    # ---------------------------------------------------------------- search
    def _bm25_leg(self, query: str, k: int, include_archived: bool, tag: Optional[str],
                  phrase: bool = False) -> List[Tuple[int, float]]:
        match = fts_query(self.tok, query, phrase=phrase)
        if not match or k <= 0:
            return []
        conds = ["memories_fts MATCH ?"]
        params: List[Any] = [match]
        join = " JOIN memories m ON m.id = memories_fts.rowid"
        if not include_archived:
            conds.append("m.archived = 0")
        if tag is not None:
            join += " JOIN memory_tags t ON t.memory_id = m.id"
            conds.append("t.tag = ?")
            params.append(tag)
        params.append(k)
        rows = self.conn.execute(
            f"SELECT memories_fts.rowid, bm25(memories_fts) AS s FROM memories_fts{join} "
            f"WHERE {' AND '.join(conds)} ORDER BY s LIMIT ?", params).fetchall()
        return [(int(i), -float(s)) for i, s in rows]  # FTS5 bm25() is negative, lower = better

    def _dense_leg(self, query: str, k: int, include_archived: bool, tag: Optional[str],
                   query_embedding: Optional[np.ndarray]) -> List[Tuple[int, float]]:
        if self.dense is None or k <= 0:
            return []
        q = query_embedding
        if q is None:
            e = self._embed([query])
            if e is None:
                return []
            q = e[0]
        allowed = None
        if tag is not None:
            allowed = {r[0] for r in self.conn.execute("SELECT memory_id FROM memory_tags WHERE tag=?", (tag,))}
        return self.dense.search(np.asarray(q, dtype=np.float32), k, include_archived, allowed)

    def _legs(self, query: str, mode: str, k: int, legs_k: Optional[int], inc: bool, tag: Optional[str],
              query_embedding: Optional[np.ndarray], phrase: bool) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        lk = legs_k or max(k * self.oversample, 10)
        dense_leg = self._dense_leg(query, lk, inc, tag, query_embedding) if mode in ("hybrid", "dense") else []
        bm25_leg = self._bm25_leg(query, lk, inc, tag, phrase) if mode in ("hybrid", "bm25") else []
        return dense_leg, bm25_leg

    def _fuse(self, mode: str, ranker: str, weights, norm: str, dense_leg, bm25_leg) -> Dict[int, float]:
        if mode == "hybrid":
            if ranker == "rrf":
                return fusion.rrf([dense_leg, bm25_leg], self.rrf_k)
            return fusion.weighted([dense_leg, bm25_leg], weights, norm)
        return dict(dense_leg) if mode == "dense" else dict(bm25_leg)

    def _before_read(self) -> None:
        """Owner: drain queued commands (which also picks up external changes).
        Reader: reload rows another connection changed since the last query."""
        if self.owner:
            self.drain_commands()
        else:
            self._check_external_changes()

    def search(self, query: str, k: int = 5, mode: str = "hybrid", ranker: str = "rrf",
               weights: Tuple[float, float] = (0.5, 0.5), norm: str = "minmax",
               archived: str = "exclude", tag: Optional[str] = None,
               query_embedding: Optional[np.ndarray] = None, legs_k: Optional[int] = None,
               phrase: bool = False, _log: bool = True) -> List[Memory]:
        """mode: hybrid | dense | bm25.  ranker: rrf (default, rank-based, no weights) |
        weighted (weights = dense/BM25 shares; norm minmax | arctan | none).
        archived: exclude | include | meta (meta = archived hits come back with text withheld
        and label 'archived').  phrase=True makes the BM25 leg an exact-substring lookup."""
        if archived not in ("exclude", "include", "meta"):
            raise ValueError("archived must be exclude | include | meta")
        t0 = time.perf_counter()
        self._before_read()
        inc = archived != "exclude"
        dense_leg, bm25_leg = self._legs(query, mode, k, legs_k, inc, tag, query_embedding, phrase)
        ranked = fusion.top(self._fuse(mode, ranker, weights, norm, dense_leg, bm25_leg), k)
        rows = self.get_many([mid for mid, _ in ranked])
        d_scores, b_scores = dict(dense_leg), dict(bm25_leg)
        out: List[Memory] = []
        for mid, s in ranked:
            m = rows.get(mid)
            if m is None or (m.superseded and archived != "include"):
                continue
            m.score = s
            m.dense = d_scores.get(mid)
            m.bm25 = b_scores.get(mid)
            if m.archived:
                m.label = "archived"
                if archived == "meta":
                    m.text = ""  # contents withheld; tags + reason carry the meta-info
            elif m.dense is not None and m.dense >= self.high:
                m.label = "high"
            out.append(m)
        if self.log_searches and _log:
            self._log_search(query, mode, out, None)
        ms = (time.perf_counter() - t0) * 1000
        if ms > self.slow_ms:
            log.warning("slow search %.1f ms (mode=%s, k=%d, n=%d): %.60s", ms, mode, k, self.count(), query)
        return out

    def explain(self, query: str, k: int = 5, mode: str = "hybrid", ranker: str = "rrf",
                weights: Tuple[float, float] = (0.5, 0.5), norm: str = "minmax", archived: str = "exclude",
                tag: Optional[str] = None, query_embedding: Optional[np.ndarray] = None,
                legs_k: Optional[int] = None, phrase: bool = False) -> Dict[str, Any]:
        """The fusion math for one query: both legs with ranks and scores, the fused list,
        the FTS5 MATCH string and the thresholds. Not logged."""
        inc = archived != "exclude"
        dense_leg, bm25_leg = self._legs(query, mode, k, legs_k, inc, tag, query_embedding, phrase)
        fused = self._fuse(mode, ranker, weights, norm, dense_leg, bm25_leg)
        d_rank, b_rank = fusion.ranks(dense_leg), fusion.ranks(bm25_leg)
        d_s, b_s = dict(dense_leg), dict(bm25_leg)
        top = fusion.top(fused, k)
        rows = self.get_many([m for m, _ in top] + [m for m, _ in dense_leg] + [m for m, _ in bm25_leg])
        text = lambda mid: (rows[mid].text if mid in rows else "?")[:80]  # noqa: E731
        return {
            "query": query, "mode": mode, "ranker": ranker, "weights": list(weights) if ranker != "rrf" else None,
            "norm": norm if ranker != "rrf" else None, "rrf_k": self.rrf_k if ranker == "rrf" else None,
            "legs_k": legs_k or max(k * self.oversample, 10), "fts_match": fts_query(self.tok, query, phrase=phrase),
            "thresholds": {"high": self.high, "drop": self.drop},
            "dense": [{"rank": r, "id": mid, "score": round(s, 4), "text": text(mid)} for r, (mid, s) in enumerate(dense_leg, 1)],
            "bm25": [{"rank": r, "id": mid, "score": round(s, 4), "text": text(mid)} for r, (mid, s) in enumerate(bm25_leg, 1)],
            "fused": [{"rank": r, "id": mid, "score": round(s, 5), "dense_rank": d_rank.get(mid), "bm25_rank": b_rank.get(mid),
                       "dense": None if mid not in d_s else round(d_s[mid], 4), "bm25": None if mid not in b_s else round(b_s[mid], 4),
                       "label": "high" if d_s.get(mid, -1) >= self.high else "plain",
                       "archived": rows[mid].archived if mid in rows else None, "text": text(mid)}
                      for r, (mid, s) in enumerate(top, 1)],
        }

    def context(self, query: str, k: int = 5, title: str = "Long-term memory", **kw) -> str:
        """Ready-to-inject block with trust labels (L3 convenience). A dense score below
        ``drop`` is filtered unless the BM25 leg also matched; archived hits appear as
        meta-info only so the agent can ask for consent instead of reciting them."""
        kw.setdefault("archived", "meta")
        hits = self.search(query, k, _log=False, **kw)
        lines = []
        injected: Set[int] = set()
        for m in hits:
            if m.label == "archived":
                topic = "/".join(t for t in m.tags if t != "archived") or "untagged"
                reason = f" — {m.archive_reason}" if m.archive_reason else ""
                lines.append(f"- [archived: {topic}{reason} — contents withheld; offer to discuss]")
                injected.add(m.id)
                continue
            if m.dense is not None and m.dense < self.drop and m.bm25 is None:
                continue
            tag_s = f" (tags: {', '.join(m.tags)})" if m.tags else ""
            lines.append(f"- {'[high] ' if m.label == 'high' else ''}{m.text}{tag_s}")
            injected.add(m.id)
        if self.log_searches:
            self._log_search(query, "context", hits, injected)
        if not lines:
            return ""
        return f"{title}:\n" + "\n".join(lines)

    def similar(self, text: str, k: int = 5, embedding: Optional[np.ndarray] = None) -> List[Memory]:
        """Candidates for dedup / conflict checks (used by L2): hybrid over active memories."""
        return self.search(text, k, mode="hybrid", query_embedding=embedding)

    # ------------------------------------------------------------------ logs
    def _log_search(self, query: str, mode: str, hits: List[Memory], injected: Optional[Set[int]]) -> None:
        now = time.time()
        with self._lock, self.conn:
            self.conn.executemany(
                "INSERT INTO search_log(at, query, mode, memory_id, rank, fused, dense, bm25, label, injected)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(now, query, mode, m.id, r, m.score, m.dense, m.bm25, m.label,
                  None if injected is None else int(m.id in injected)) for r, m in enumerate(hits, 1)])

    def search_log(self, limit: int = 1000, since: Optional[float] = None) -> List[Dict[str, Any]]:
        q = f"SELECT {', '.join(_LOG_COLS)} FROM search_log{' WHERE at >= ?' if since else ''} ORDER BY id DESC LIMIT ?"
        params: List[Any] = ([since] if since else []) + [limit]
        return [dict(zip(_LOG_COLS, r)) for r in self.conn.execute(q, params)]

    def log_pipeline_run(self, user_text: str, candidates: Sequence[int], raw: str, applied: Any, skipped: Any,
                         error: Optional[str], model: Optional[str], ms: float) -> None:
        if not self.log_pipeline:
            return
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO pipeline_log(at, user_text, candidates, raw, applied, skipped, error, model, ms)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), user_text, json.dumps(list(candidates)), raw, json.dumps(applied, ensure_ascii=False),
                 json.dumps(skipped, ensure_ascii=False), error, model, ms))

    def pipeline_log_rows(self, limit: int = 100, since: Optional[float] = None) -> List[Dict[str, Any]]:
        q = f"SELECT {', '.join(_PLOG_COLS)} FROM pipeline_log{' WHERE at >= ?' if since else ''} ORDER BY id DESC LIMIT ?"
        params: List[Any] = ([since] if since else []) + [limit]
        out = []
        for r in self.conn.execute(q, params):
            d = dict(zip(_PLOG_COLS, r))
            for key in ("candidates", "applied", "skipped"):
                try:
                    d[key] = json.loads(d[key]) if d[key] else []
                except Exception:
                    pass
            out.append(d)
        return out

    def prune_logs(self, days: int) -> int:
        cutoff = time.time() - days * 86400
        with self._lock, self.conn:
            n = self.conn.execute("DELETE FROM search_log WHERE at < ?", (cutoff,)).rowcount
            n += self.conn.execute("DELETE FROM pipeline_log WHERE at < ?", (cutoff,)).rowcount
            n += self.conn.execute("DELETE FROM commands WHERE done_at IS NOT NULL AND done_at < ?", (cutoff,)).rowcount
        return int(n)

    def score_percentiles(self, field: str = "dense", since: Optional[float] = None,
                          injected_only: bool = False, label: Optional[str] = None) -> Dict[str, float]:
        """Distribution of logged scores — the input for setting ``high``/``drop``."""
        rows = self.search_log(limit=10 ** 9, since=since)
        vals = sorted(r[field] for r in rows if r[field] is not None
                      and (not injected_only or r["injected"]) and (label is None or r["label"] == label))
        if not vals:
            return {}
        pick = lambda p: vals[min(len(vals) - 1, int(round(p / 100 * (len(vals) - 1))))]  # noqa: E731
        return {"n": len(vals), "min": vals[0], "p10": pick(10), "p25": pick(25), "p50": pick(50),
                "p75": pick(75), "p90": pick(90), "max": vals[-1]}

    # ------------------------------------------------- ownership & commands
    def owner_info(self) -> Optional[Dict[str, Any]]:
        raw = self._meta("owner")
        return json.loads(raw) if raw else None

    def owner_alive(self, max_age: float = 5.0) -> bool:
        info = self.owner_info()
        return bool(info) and (time.time() - float(info.get("seen", 0))) < max_age

    def _heartbeat(self, force: bool = False) -> None:
        now = time.time()
        if force or now - self._last_heartbeat > 2.0:
            self._set_meta("owner", json.dumps({"pid": os.getpid(), "seen": now}))
            self._last_heartbeat = now

    def start_owner_loop(self, interval: float = 1.0) -> None:
        """Drain the command queue on a timer (so deletes land even while she is idle)."""
        if not self.owner or self._owner_thread:
            return

        def run() -> None:
            while not self._stop.wait(interval):
                try:
                    self.drain_commands()
                except Exception:
                    log.exception("owner loop")

        self._owner_thread = threading.Thread(target=run, daemon=True, name="keepsake-owner")
        self._owner_thread.start()

    def enqueue(self, op: str, payload: Optional[Dict[str, Any]] = None, by: str = "") -> int:
        if op not in COMMAND_OPS:
            raise ValueError(f"unknown command {op!r} (one of {COMMAND_OPS})")
        with self._lock, self.conn:
            cur = self.conn.execute("INSERT INTO commands(op, payload, requested_at, requested_by) VALUES (?, ?, ?, ?)",
                                    (op, json.dumps(payload or {}, ensure_ascii=False), time.time(), by))
        return int(cur.lastrowid)

    def pending_commands(self) -> List[Dict[str, Any]]:
        return [dict(zip(_CMD_COLS, r)) for r in self.conn.execute(
            f"SELECT {', '.join(_CMD_COLS)} FROM commands WHERE done_at IS NULL ORDER BY id")]

    def wait_command(self, cmd_id: int, timeout: float = 10.0, poll: float = 0.2) -> Optional[Dict[str, Any]]:
        """Block until the owner has executed the command; None on timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            r = self.conn.execute("SELECT done_at, result FROM commands WHERE id=?", (cmd_id,)).fetchone()
            if r and r[0] is not None:
                return json.loads(r[1]) if r[1] else {}
            time.sleep(poll)
        return None

    def drain_commands(self) -> int:
        """Owner only: execute queued commands in order. Also the owner's heartbeat and
        its external-change check."""
        if not self.owner:
            raise RuntimeError("drain_commands is for the owner process")
        with self._lock:
            self._check_external_changes()
            self._heartbeat()
            n = 0
            for cmd in self.pending_commands():
                try:
                    result = self._run_command(cmd["op"], json.loads(cmd["payload"] or "{}"))
                except Exception as e:
                    result = {"error": f"{type(e).__name__}: {e}"}
                    log.exception("command %s failed", cmd["op"])
                with self.conn:
                    self.conn.execute("UPDATE commands SET done_at=?, result=? WHERE id=?",
                                      (time.time(), json.dumps(result, ensure_ascii=False), cmd["id"]))
                log.info("command %s %s -> %s", cmd["id"], cmd["op"], result)
                n += 1
            return n

    def _run_command(self, op: str, p: Dict[str, Any]) -> Dict[str, Any]:
        if op == "ping":
            return {"ok": True, "pid": os.getpid()}
        if op in ("delete", "archive", "unarchive"):
            fn = {"delete": self.delete, "archive": self.archive, "unarchive": self.unarchive}[op]
            n = fn(ids=p.get("ids"), tag=p.get("tag"), reason=p.get("reason", ""))
            return {"count": n}
        if op == "backup":
            return {"path": self.snapshot(p.get("dir"), text_only=p.get("text_only"), reason=p.get("reason", "command"))}
        if op == "restore":
            self.restore(p["path"])
            return {"restored": p["path"], "memories": self.count(include_archived=True)}
        if op == "reindex":
            return {"reindexed": self.reindex(p.get("tokenizer", self.tok.name))}
        if op == "reembed":
            if self.embedder is None:
                raise RuntimeError("owner has no embedder")
            return {"reembedded": self.reembed(self.embedder)}
        raise ValueError(op)

    def _check_external_changes(self) -> None:
        """PRAGMA data_version changes only when ANOTHER connection commits. Reload the
        rows named in history since our last look (delete -> drop, else re-read)."""
        v = self._pragma_data_version()
        if v == self._data_version:
            return
        self._data_version = v
        rows = self.conn.execute("SELECT DISTINCT memory_id FROM history WHERE id > ?", (self._last_history_id,)).fetchall()
        self._last_history_id = self._max_history_id()
        if not rows or self.dense is None or self._dense_kind != "numpy":
            return
        for (mid,) in rows:
            r = self.conn.execute("SELECT embedding, archived FROM memories WHERE id=?", (mid,)).fetchone()
            if r is None:
                self.dense.remove(mid)
            elif r[0] is not None:
                self.dense.add(mid, np.frombuffer(r[0], dtype=np.float32), ARCHIVED if r[1] else ACTIVE)
        log.info("reloaded %d memories changed by another connection", len(rows))

    # --------------------------------------------------------------- backups
    def backup(self, dest: str, include_embeddings: bool = True) -> str:
        """Consistent copy of the live database (SQLite online backup API), written to a temp
        name and renamed into place. include_embeddings=False drops the vectors (≈ 80 % of the
        file); restoring such a snapshot re-embeds."""
        tmp = dest + ".tmp"
        if os.path.exists(tmp):
            os.remove(tmp)
        os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
        dst = sqlite3.connect(tmp)
        try:
            with self._lock:
                self.conn.backup(dst)
            if not include_embeddings:
                dst.execute("UPDATE memories SET embedding=NULL")
                dst.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('text_only', '1')")
                dst.execute("DELETE FROM meta WHERE key='owner'")
                dst.commit()
                dst.execute("VACUUM")
            else:
                dst.execute("DELETE FROM meta WHERE key='owner'")
                dst.commit()
        finally:
            dst.close()
        os.replace(tmp, dest)
        log.info("backup -> %s (%s)", dest, "text-only" if not include_embeddings else "full")
        return dest

    def snapshot(self, folder: Optional[str] = None, text_only: Optional[bool] = None, reason: str = "",
                 keep_daily: int = 7, keep_weekly: int = 4, keep_monthly: int = 12) -> str:
        """Timestamped snapshot in ``folder`` (default backup_dir, else <db>.backups/) + rotation."""
        folder = folder or self.backup_dir or (os.path.splitext(self.path)[0] + ".backups")
        text_only = self.snapshot_text_only if text_only is None else bool(text_only)
        stem = os.path.splitext(os.path.basename(self.path))[0]
        name = f"{stem}-{datetime.now().strftime('%Y%m%d-%H%M%S')}{'-text' if text_only else ''}.db"
        dest = self.backup(os.path.join(folder, name), include_embeddings=not text_only)
        with self._lock, self.conn:
            self._set_meta("last_snapshot", json.dumps({"path": dest, "at": time.time(), "reason": reason}))
            self._set_meta("last_snapshot_day", datetime.now().strftime("%Y-%m-%d"))
        self.rotate(folder, keep_daily, keep_weekly, keep_monthly)
        return dest

    def rotate(self, folder: str, keep_daily: int = 7, keep_weekly: int = 4, keep_monthly: int = 12) -> List[str]:
        """Keep the newest snapshot per day for keep_daily days, per ISO week for keep_weekly
        weeks, per month for keep_monthly months (Anki-style); delete the rest."""
        stem = os.path.splitext(os.path.basename(self.path))[0]
        snaps = []
        for f in os.listdir(folder) if os.path.isdir(folder) else []:
            m = _SNAP_RE.match(f)
            if m and m.group("stem") == stem:
                snaps.append((datetime.strptime(m.group("ts"), "%Y%m%d-%H%M%S"), f))
        snaps.sort(reverse=True)
        if not snaps:
            return []
        now = snaps[0][0]
        keep: Set[str] = {snaps[0][1]}
        seen_day: Set[str] = set()
        seen_week: Set[str] = set()
        seen_month: Set[str] = set()
        for ts, f in snaps:
            age_days = (now - ts).days
            day, week, month = ts.strftime("%Y-%m-%d"), ts.strftime("%G-W%V"), ts.strftime("%Y-%m")
            if age_days < keep_daily and day not in seen_day:
                seen_day.add(day)
                keep.add(f)
            elif age_days < keep_weekly * 7 and week not in seen_week:
                seen_week.add(week)
                keep.add(f)
            elif age_days < keep_monthly * 31 and month not in seen_month:
                seen_month.add(month)
                keep.add(f)
        deleted = []
        for _, f in snaps:
            if f not in keep:
                os.remove(os.path.join(folder, f))
                deleted.append(f)
        return deleted

    def _auto_snapshot(self, force: bool = False, reason: str = "daily") -> Optional[str]:
        if not self.backup_dir:
            return None
        today = datetime.now().strftime("%Y-%m-%d")
        if not force and self._meta("last_snapshot_day") == today:
            return None
        try:
            return self.snapshot(reason=reason)
        except Exception:
            log.exception("auto snapshot failed")
            return None

    def restore(self, snapshot_path: str) -> None:
        """Replace the live contents with a snapshot (online backup API into the open
        connection, so readers never see a torn file). Takes a snapshot of the current state
        first when backup_dir is set. A text-only snapshot is re-embedded if an embedder is
        attached, otherwise the dense leg stays empty until ``reembed``."""
        if not os.path.exists(snapshot_path):
            raise FileNotFoundError(snapshot_path)
        self._auto_snapshot(force=True, reason="before restore")
        src = sqlite3.connect(snapshot_path)
        try:
            with self._lock:
                src.backup(self.conn)
        finally:
            src.close()
        with self._lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("DELETE FROM meta WHERE key='owner'")
            stored = self._meta("dims")
            self.dims = int(stored) if stored else self.dims
            if self.dims:
                self._init_dense()
            self._data_version = self._pragma_data_version()
            self._last_history_id = self._max_history_id()
            text_only = self._meta("text_only") == "1"
        log.info("restored %s (%d memories%s)", snapshot_path, self.count(include_archived=True),
                 ", text-only" if text_only else "")
        if text_only and self.embedder is not None:
            self.reembed(self.embedder)
            with self._lock, self.conn:
                self.conn.execute("DELETE FROM meta WHERE key='text_only'")

    # ------------------------------------------------------------ maintenance
    def reembed(self, embedder, batch: int = 64) -> int:
        """Switch embedder: re-embed every memory (minutes at companion scale)."""
        self._auto_snapshot(force=True, reason="before reembed")
        self.embedder = embedder
        rows = self.conn.execute("SELECT id, text, archived FROM memories ORDER BY id").fetchall()
        self.dims = None
        n = 0
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            vecs = self._embed([t for _, t, _ in chunk])
            with self._lock, self.conn:
                for (mid, _, arch), v in zip(chunk, vecs):
                    self.conn.execute("UPDATE memories SET embedding=?, embedder=? WHERE id=?",
                                      (v.astype(np.float32).tobytes(), embedder.name, mid))
                    self.dense.add(mid, v, ARCHIVED if arch else ACTIVE)
                    n += 1
        with self._lock, self.conn:
            self._set_meta("last_embedder", embedder.name)
        log.info("reembedded %d memories with %s", n, embedder.name)
        return n

    def reindex(self, tokenizer: "str | Tokenizer") -> int:
        """Switch tokenizer: rebuild the FTS5 index."""
        self._auto_snapshot(force=True, reason="before reindex")
        self.tok = get_tokenizer(tokenizer)
        rows = self.conn.execute("SELECT id, text FROM memories").fetchall()
        with self._lock, self.conn:
            self.conn.execute("DELETE FROM memories_fts")
            for mid, text in rows:
                toks = index_string(self.tok, text)
                self.conn.execute("UPDATE memories SET tokens=? WHERE id=?", (toks, mid))
                self.conn.execute("INSERT INTO memories_fts(rowid, tokens) VALUES (?, ?)", (mid, toks))
            self._set_meta("tokenizer", self.tok.name)
        log.info("reindexed %d memories with %s", len(rows), self.tok.name)
        return len(rows)

    def optimize(self) -> None:
        self.conn.execute("INSERT INTO memories_fts(memories_fts) VALUES ('optimize')")
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def close(self) -> None:
        self._stop.set()
        if self._owner_thread:
            self._owner_thread.join(timeout=2)
            self._owner_thread = None
        if self.owner:
            try:
                with self.conn:
                    self.conn.execute("DELETE FROM meta WHERE key='owner'")
            except Exception:
                pass
        self.conn.close()
        log.info("closed %s", self.path)

    def __enter__(self) -> "Keepsake":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
