"""``keepsake`` — inspect and maintain a store from the terminal (for people not running 556,
and for our own testing). Write operations go through the owner's command queue when an
owner process is alive, and straight to the file otherwise.

    keepsake --db her.db stats
    keepsake --db her.db search "我的貓" --explain [--embedder http://127.0.0.1:55602/v1]
    keepsake --db her.db delete 42 --reason "user asked"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, List, Optional

from .store import Keepsake


def _fmt_ts(t: Optional[float]) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) if t else "-"


def _out(args, obj: Any, text: str) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=1, default=str) if args.json else text)


def _mem_line(m) -> str:
    state = "superseded" if m.superseded else ("archived" if m.archived else "active")
    extra = f" [{state}: {m.archive_reason or ''}]" if m.archived else ""
    tags = f" ({', '.join(m.tags)})" if m.tags else ""
    return f"#{m.id:<5} {m.text}{tags}{extra}"


def _write_op(ks: Keepsake, args, op: str, payload: dict) -> dict:
    """Route through the owner if one is alive (keeps its dense cache coherent)."""
    if ks.owner_alive():
        cid = ks.enqueue(op, payload, by="cli")
        res = ks.wait_command(cid, timeout=args.timeout)
        if res is None:
            return {"queued": cid, "note": "owner did not answer in time; it will run when the owner drains"}
        return res
    fn = {"delete": ks.delete, "archive": ks.archive, "unarchive": ks.unarchive}[op]
    return {"count": fn(ids=payload.get("ids"), tag=payload.get("tag"), reason=payload.get("reason", "")), "direct": True}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="keepsake", description="KeepsakeDB store inspector")
    p.add_argument("--db", default=os.environ.get("KEEPSAKE_DB"), required=os.environ.get("KEEPSAKE_DB") is None,
                   help="store file (or env KEEPSAKE_DB)")
    p.add_argument("--embedder", default=os.environ.get("KEEPSAKE_EMBEDDER"),
                   help="OpenAI-compatible /v1 base URL for the dense leg (or env KEEPSAKE_EMBEDDER)")
    p.add_argument("--model", default=os.environ.get("KEEPSAKE_EMBED_MODEL", "bge-m3"))
    p.add_argument("--tokenizer", default=None, help="only needed if the store is not bigram")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--timeout", type=float, default=10.0, help="seconds to wait for the owner")
    # --json is also accepted after the subcommand (SUPPRESS keeps the sub-level default from
    # overriding a --json given before it)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = lambda name: sub.add_parser(name, parents=[common])  # noqa: E731

    sp("stats")
    s = sp("list"); s.add_argument("--tag"); s.add_argument("--archived", action="store_true"); s.add_argument("--limit", type=int, default=50)
    s = sp("show"); s.add_argument("id", type=int)
    s = sp("search"); s.add_argument("query"); s.add_argument("-k", type=int, default=5)
    s.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default="hybrid"); s.add_argument("--tag")
    s.add_argument("--phrase", action="store_true"); s.add_argument("--explain", action="store_true")
    s.add_argument("--archived", choices=["exclude", "include", "meta"], default="exclude")
    sp("tags")
    s = sp("log"); s.add_argument("--limit", type=int, default=20)
    s = sp("plog"); s.add_argument("--limit", type=int, default=10)
    for name in ("delete", "archive", "unarchive"):
        s = sp(name); s.add_argument("ids", type=int, nargs="*"); s.add_argument("--tag"); s.add_argument("--reason", default="")
    s = sp("backup"); s.add_argument("--dir"); s.add_argument("--text-only", action="store_true")
    s = sp("restore"); s.add_argument("path")
    sp("commands")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    embedder = None
    if args.embedder:
        from .embedders import OpenAIEmbedder
        embedder = OpenAIEmbedder(args.embedder, args.model)
    tok = args.tokenizer
    if tok is None:  # read the store's tokenizer so we never fail on mismatch
        import sqlite3
        try:
            c = sqlite3.connect(args.db)
            r = c.execute("SELECT value FROM meta WHERE key='tokenizer'").fetchone()
            c.close()
            tok = r[0] if r else "bigram"
        except Exception:
            tok = "bigram"
    ks = Keepsake(args.db, embedder=embedder, tokenizer=tok, log_searches=False, log_pipeline=False,
                  log_retention_days=None)
    try:
        return _run(ks, args)
    finally:
        ks.close()


def _run(ks: Keepsake, args) -> int:
    c = args.cmd
    if c == "stats":
        st = ks.stats()
        _out(args, st, "\n".join(f"{k:18s} {v}" for k, v in st.items()))
    elif c == "list":
        rows = ks.by_tag(args.tag, include_archived=args.archived) if args.tag else ks.all(include_archived=args.archived, limit=args.limit)
        rows = rows[: args.limit]
        _out(args, [m.to_dict() for m in rows], "\n".join(_mem_line(m) for m in rows) or "(empty)")
    elif c == "show":
        m = ks.get(args.id)
        if m is None:
            print(f"no memory #{args.id}")
            return 1
        hist = ks.history(args.id)
        _out(args, {"memory": m.to_dict(), "history": hist},
             _mem_line(m) + f"\n  created {_fmt_ts(m.created_at)}  updated {_fmt_ts(m.updated_at)}  meta {m.meta}\n" +
             "\n".join(f"  {_fmt_ts(h['at'])} {h['op']:10s} {h.get('reason') or ''} {('| ' + h['new_text']) if h.get('new_text') else ''}" for h in hist))
    elif c == "search":
        mode = args.mode
        if mode != "bm25" and ks.embedder is None and ks.dense is None:
            mode = "bm25"
        elif mode != "bm25" and ks.embedder is None:
            print("(no --embedder given: dense leg skipped, BM25 only)")
            mode = "bm25"
        if args.explain:
            ex = ks.explain(args.query, k=args.k, mode=mode, archived=args.archived, tag=args.tag, phrase=args.phrase)
            if args.json:
                _out(args, ex, "")
            else:
                print(f"query: {ex['query']}  mode={ex['mode']} ranker={ex['ranker']} rrf_k={ex['rrf_k']} legs_k={ex['legs_k']}")
                print(f"fts match: {ex['fts_match']}")
                for leg in ("dense", "bm25"):
                    print(f"-- {leg} leg")
                    for r in ex[leg]:
                        print(f"  {r['rank']:>2}. #{r['id']:<5} {r['score']:8.4f}  {r['text']}")
                print("-- fused")
                for r in ex["fused"]:
                    print(f"  {r['rank']:>2}. #{r['id']:<5} {r['score']:8.5f}  d#{r['dense_rank'] or '-'} b#{r['bm25_rank'] or '-'}  "
                          f"dense={r['dense']} bm25={r['bm25']} [{r['label']}{', archived' if r['archived'] else ''}]  {r['text']}")
        else:
            hits = ks.search(args.query, k=args.k, mode=mode, archived=args.archived, tag=args.tag, phrase=args.phrase, _log=False)
            _out(args, [m.to_dict() for m in hits],
                 "\n".join(f"{i}. {_mem_line(m)}  fused={m.score:.4f} dense={m.dense} bm25={m.bm25} [{m.label}]"
                           for i, m in enumerate(hits, 1)) or "(no hits)")
    elif c == "tags":
        t = ks.tags(include_archived=True)
        _out(args, t, "\n".join(f"{k:20s} {v}" for k, v in t.items()) or "(no tags)")
    elif c == "log":
        rows = ks.search_log(limit=args.limit * 10)
        by_q: dict = {}
        for r in rows:
            by_q.setdefault((r["at"], r["query"], r["mode"]), []).append(r)
        lines = []
        for (at, q, mode), hits in list(by_q.items())[: args.limit]:
            lines.append(f"{_fmt_ts(at)} [{mode}] {q}")
            for h in sorted(hits, key=lambda x: x["rank"]):
                inj = "" if h["injected"] is None else (" INJECTED" if h["injected"] else " dropped")
                lines.append(f"    {h['rank']}. #{h['memory_id']} fused={h['fused']:.4f} dense={h['dense']} bm25={h['bm25']} [{h['label']}]{inj}")
        _out(args, rows, "\n".join(lines) or "(no searches logged)")
    elif c == "plog":
        rows = ks.pipeline_log_rows(limit=args.limit)
        lines = []
        for r in rows:
            lines.append(f"{_fmt_ts(r['at'])} {r['ms']:.0f} ms {r['model'] or ''} | {r['user_text'][:70]}")
            lines.append(f"    candidates {r['candidates']}  applied {r['applied']}  skipped {r['skipped']}  error {r['error'] or '-'}")
        _out(args, rows, "\n".join(lines) or "(no pipeline runs logged)")
    elif c in ("delete", "archive", "unarchive"):
        if not args.ids and not args.tag:
            print("give ids or --tag")
            return 2
        res = _write_op(ks, args, c, {"ids": args.ids or None, "tag": args.tag, "reason": args.reason})
        _out(args, res, f"{c}: {res}")
    elif c == "backup":
        if ks.owner_alive():
            cid = ks.enqueue("backup", {"dir": args.dir, "text_only": args.text_only, "reason": "cli"}, by="cli")
            res = ks.wait_command(cid, timeout=max(args.timeout, 60)) or {"queued": cid}
        else:
            res = {"path": ks.snapshot(args.dir, text_only=args.text_only, reason="cli"), "direct": True}
        _out(args, res, f"backup: {res}")
    elif c == "restore":
        if ks.owner_alive():
            cid = ks.enqueue("restore", {"path": args.path}, by="cli")
            res = ks.wait_command(cid, timeout=max(args.timeout, 120)) or {"queued": cid}
        else:
            ks.restore(args.path)
            res = {"restored": args.path, "memories": ks.count(include_archived=True), "direct": True}
        _out(args, res, f"restore: {res}")
    elif c == "commands":
        rows = ks.pending_commands()
        _out(args, rows, "\n".join(f"#{r['id']} {r['op']} {r['payload']} by {r['requested_by']} at {_fmt_ts(r['requested_at'])}" for r in rows) or "(queue empty)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
