"""Dense leg: exact cosine top-k (what Milvus calls a FLAT index).

At companion scale (<= ~100k memories) brute force on a normalized float32 matrix is
a few milliseconds and never loses recall, so an ANN index (HNSW/IVF) would only add
build time and approximation. Two backends:

- NumpyDense (default): matrix in RAM, loaded from SQLite at open. 1024-d float32 is
  4 KB per memory (10k memories = 40 MB).
- SqliteVecDense: a sqlite-vec ``vec0`` virtual table inside the same database file;
  on-disk brute force with metadata filtering, no RAM matrix. ``pip install keepsakedb[vec]``.
"""
from __future__ import annotations

import sqlite3
from typing import Iterable, List, Optional, Set, Tuple

import numpy as np

ACTIVE, ARCHIVED, DELETED = 0, 1, 2


class NumpyDense:
    name = "numpy"

    def __init__(self, dims: int, dtype=np.float32) -> None:
        self.dims = dims
        self.dtype = dtype
        self._cap = 1024
        self._n = 0
        self._mat = np.zeros((self._cap, dims), dtype=dtype)
        self._ids = np.zeros(self._cap, dtype=np.int64)
        self._flags = np.zeros(self._cap, dtype=np.int8)
        self._pos: dict = {}

    # -- maintenance --------------------------------------------------------
    def load(self, rows: Iterable[Tuple[int, bytes, int]]) -> None:
        for mid, blob, flag in rows:
            self.add(mid, np.frombuffer(blob, dtype=np.float32), flag)

    def add(self, mid: int, vec: np.ndarray, flag: int = ACTIVE) -> None:
        if mid in self._pos:  # re-embed / update in place
            p = self._pos[mid]
            self._mat[p] = vec.astype(self.dtype)
            self._flags[p] = flag
            return
        if self._n == self._cap:
            new_cap = self._cap * 2
            mat = np.zeros((new_cap, self.dims), dtype=self.dtype)
            mat[: self._n] = self._mat[: self._n]
            self._mat = mat
            self._ids = np.concatenate([self._ids, np.zeros(new_cap - self._cap, dtype=np.int64)])
            self._flags = np.concatenate([self._flags, np.zeros(new_cap - self._cap, dtype=np.int8)])
            self._cap = new_cap
        p = self._n
        self._mat[p] = vec.astype(self.dtype)
        self._ids[p] = mid
        self._flags[p] = flag
        self._pos[mid] = p
        self._n += 1

    def set_flag(self, mid: int, flag: int) -> None:
        p = self._pos.get(mid)
        if p is not None:
            self._flags[p] = flag

    def remove(self, mid: int) -> None:
        self.set_flag(mid, DELETED)

    # -- query --------------------------------------------------------------
    def search(self, q: np.ndarray, k: int, include_archived: bool = False,
               allowed: Optional[Set[int]] = None) -> List[Tuple[int, float]]:
        if self._n == 0 or k <= 0:
            return []
        mat = self._mat[: self._n]
        scores = mat @ q.astype(self.dtype)
        flags = self._flags[: self._n]
        mask = (flags == ACTIVE) if not include_archived else (flags != DELETED)
        if allowed is not None:
            mask &= np.isin(self._ids[: self._n], np.fromiter(allowed, dtype=np.int64, count=len(allowed)))
        scores = np.where(mask, scores, -np.inf)
        k = min(k, int(mask.sum()))
        if k <= 0:
            return []
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(self._ids[i]), float(scores[i])) for i in idx if np.isfinite(scores[i])]

    def __len__(self) -> int:
        return int((self._flags[: self._n] != DELETED).sum())


class SqliteVecDense:
    name = "sqlite-vec"

    def __init__(self, conn: sqlite3.Connection, dims: int, table: str = "memories_vec") -> None:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        self.conn = conn
        self.dims = dims
        self.table = table
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0("
            f"id INTEGER PRIMARY KEY, embedding float[{dims}] distance_metric=cosine, flag INTEGER)"
        )

    def load(self, rows: Iterable[Tuple[int, bytes, int]]) -> None:
        pass  # persistent; nothing to load

    def add(self, mid: int, vec: np.ndarray, flag: int = ACTIVE) -> None:
        blob = np.asarray(vec, dtype=np.float32).tobytes()
        self.conn.execute(f"DELETE FROM {self.table} WHERE id = ?", (mid,))
        self.conn.execute(f"INSERT INTO {self.table}(id, embedding, flag) VALUES (?, ?, ?)", (mid, blob, flag))

    def set_flag(self, mid: int, flag: int) -> None:
        self.conn.execute(f"UPDATE {self.table} SET flag = ? WHERE id = ?", (flag, mid))

    def remove(self, mid: int) -> None:
        self.conn.execute(f"DELETE FROM {self.table} WHERE id = ?", (mid,))

    def search(self, q: np.ndarray, k: int, include_archived: bool = False,
               allowed: Optional[Set[int]] = None) -> List[Tuple[int, float]]:
        if k <= 0:
            return []
        blob = np.asarray(q, dtype=np.float32).tobytes()
        cond = "flag != 2" if include_archived else "flag = 0"
        kk = k if allowed is None else max(k * 4, 64)
        rows = self.conn.execute(
            f"SELECT id, distance FROM {self.table} WHERE embedding MATCH ? AND k = ? AND {cond} ORDER BY distance",
            (blob, kk),
        ).fetchall()
        out = [(int(i), 1.0 - float(d)) for i, d in rows if allowed is None or int(i) in allowed]
        return out[:k]

    def __len__(self) -> int:
        return int(self.conn.execute(f"SELECT count(*) FROM {self.table} WHERE flag != 2").fetchone()[0])
