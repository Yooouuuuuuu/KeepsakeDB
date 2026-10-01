"""L1 — the store. One SQLite file holding memories, tags, archive state, history,
embeddings, an FTS5 BM25 index and (optionally) a sqlite-vec index.

Hybrid search = dense cosine leg + BM25 leg -> RRF (or weighted) fusion, the same
shape as Milvus ``hybrid_search`` with ``RRFRanker``. Unlike Milvus Lite, BM25
statistics are global: FTS5 keeps one index for the whole table, so IDF/avgdl never
depend on which flush segment a memory landed in.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from . import fusion
from .dense import ACTIVE, ARCHIVED, NumpyDense, SqliteVecDense
from .tokenize import Tokenizer, fts_query, get_tokenizer, index_string

SCHEMA_VERSION = 1

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
"""

_COLS = "id, text, tags, meta, archived, archive_reason, created_at, updated_at"


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
                 oversample: int = 3) -> None:
        self.path = path
        self.embedder = embedder
        self.tok = get_tokenizer(tokenizer)
        self.rrf_k = rrf_k
        self.high = high
        self.drop = drop
        self.oversample = oversample
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
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

    # ---------------------------------------------------------------- meta
    def _meta(self, key: str) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))

    def _set_meta_default(self, key: str, value: str) -> None:
        self.conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)", (key, value))

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
        return len(targets)

    def delete(self, ids: Optional[Iterable[int]] = None, tag: Optional[str] = None, reason: str = "") -> int:
        """Hard delete (user-commanded purge). History keeps the text."""
        targets = self._ids_for(ids, tag)
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

    def all(self, include_archived: bool = True) -> List[Memory]:
        cond = "" if include_archived else " WHERE archived=0"
        return [self._row(r) for r in self.conn.execute(f"SELECT {_COLS} FROM memories{cond} ORDER BY id")]

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

    def search(self, query: str, k: int = 5, mode: str = "hybrid", ranker: str = "rrf",
               weights: Tuple[float, float] = (0.5, 0.5), norm: str = "arctan",
               archived: str = "exclude", tag: Optional[str] = None,
               query_embedding: Optional[np.ndarray] = None, legs_k: Optional[int] = None,
               phrase: bool = False) -> List[Memory]:
        """mode: hybrid | dense | bm25.  archived: exclude | include | meta
        (meta = archived hits come back with text withheld and label 'archived').
        phrase=True makes the BM25 leg an exact-substring lookup (names, keywords)."""
        if archived not in ("exclude", "include", "meta"):
            raise ValueError("archived must be exclude | include | meta")
        inc = archived != "exclude"
        lk = legs_k or max(k * self.oversample, 10)
        dense_leg = self._dense_leg(query, lk, inc, tag, query_embedding) if mode in ("hybrid", "dense") else []
        bm25_leg = self._bm25_leg(query, lk, inc, tag, phrase) if mode in ("hybrid", "bm25") else []
        if mode == "hybrid":
            if ranker == "rrf":
                fused = fusion.rrf([dense_leg, bm25_leg], self.rrf_k)
            else:
                fused = fusion.weighted([dense_leg, bm25_leg], weights, norm)
        elif mode == "dense":
            fused = dict(dense_leg)
        else:
            fused = dict(bm25_leg)
        ranked = fusion.top(fused, k)
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
        return out

    def context(self, query: str, k: int = 5, title: str = "Long-term memory", **kw) -> str:
        """Ready-to-inject block with trust labels (L3 convenience). A dense score below
        ``drop`` is filtered unless the BM25 leg also matched; archived hits appear as
        meta-info only so the agent can ask for consent instead of reciting them."""
        kw.setdefault("archived", "meta")
        hits = self.search(query, k, **kw)
        lines = []
        for m in hits:
            if m.label == "archived":
                topic = "/".join(t for t in m.tags if t != "archived") or "untagged"
                reason = f" — {m.archive_reason}" if m.archive_reason else ""
                lines.append(f"- [archived: {topic}{reason} — contents withheld; offer to discuss]")
                continue
            if m.dense is not None and m.dense < self.drop and m.bm25 is None:
                continue
            tag_s = f" (tags: {', '.join(m.tags)})" if m.tags else ""
            lines.append(f"- {'[high] ' if m.label == 'high' else ''}{m.text}{tag_s}")
        if not lines:
            return ""
        return f"{title}:\n" + "\n".join(lines)

    def similar(self, text: str, k: int = 5, embedding: Optional[np.ndarray] = None) -> List[Memory]:
        """Candidates for dedup / conflict checks (used by L2): hybrid over active memories."""
        return self.search(text, k, mode="hybrid", query_embedding=embedding)

    # ------------------------------------------------------------ maintenance
    def reembed(self, embedder, batch: int = 64) -> int:
        """Switch embedder: re-embed every memory (minutes at companion scale)."""
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
        return n

    def reindex(self, tokenizer: "str | Tokenizer") -> int:
        """Switch tokenizer: rebuild the FTS5 index."""
        self.tok = get_tokenizer(tokenizer)
        rows = self.conn.execute("SELECT id, text FROM memories").fetchall()
        with self._lock, self.conn:
            self.conn.execute("DELETE FROM memories_fts")
            for mid, text in rows:
                toks = index_string(self.tok, text)
                self.conn.execute("UPDATE memories SET tokens=? WHERE id=?", (toks, mid))
                self.conn.execute("INSERT INTO memories_fts(rowid, tokens) VALUES (?, ?)", (mid, toks))
            self._set_meta("tokenizer", self.tok.name)
        return len(rows)

    def optimize(self) -> None:
        self.conn.execute("INSERT INTO memories_fts(memories_fts) VALUES ('optimize')")
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Keepsake":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
