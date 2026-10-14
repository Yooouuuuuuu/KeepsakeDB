"""L2 — the memory pipeline. Pluggable: nothing in L1 depends on it.

observe(turn)      one LLM pass decides add / update / supersede / archive / noop, with the
                   most similar memories and the tag groups as context (mem0's ADD/UPDATE/DELETE
                   pattern + the C2 tag-organized, conflict-aware merge).
context_for(text)  the injection block; archived groups show as meta-info only, unless the
                   user granted consent (a temp window of N turns, capped at session end).
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..llms import LLM, extract_json
from ..store import Keepsake, Memory
from . import prompts

_TAG_RE = re.compile(r"[^a-z0-9_-]+")


def _slug(t: str) -> str:
    return _TAG_RE.sub("-", str(t).strip().lower()).strip("-")[:32]


@dataclass
class PipelineConfig:
    language: str = "zh-TW"       # memory language; the embedder must cover it
    max_candidates: int = 8       # similar memories shown to the LLM
    consent_turns: int = 100      # how long a granted consent window lasts
    context_k: int = 5
    prefilter: Optional[Callable[[str], bool]] = None  # return False to skip the LLM (regex, length...)
    user_facts_only: bool = True
    migrate_subject: bool = False  # True: a subject different from the store's record rewrites every memory (the user changed the word)
    subject: Optional[str] = None  # fixed subject word for memories about the user (e.g. 你); recorded in the store
    render_subject: Optional[str] = None  # how the subject reads when injected (their name / address term); None = as stored
    user_name: Optional[str] = None  # the user's current name (prompt-layer only; never the stored subject)
    retries: int = 1              # extra attempts when the LLM reply is empty / not JSON (transient backend glitches)
    retry_delay_s: float = 2.0
    short_layer: bool = True      # the same observe call may return a dated note for the short layer
    short_k: int = 5              # notes injected per turn (long facts use context_k)
    short_cap: int = 1000         # ring size of the short layer (oldest dropped)
    surface: str = ""             # stamped on notes (meta), e.g. 事務所 / Telegram — never part of the text


@dataclass
class ObserveResult:
    note: Optional[str] = None
    applied: List[Dict[str, Any]] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)
    raw: str = ""
    error: Optional[str] = None


class Pipeline:
    def __init__(self, store: Keepsake, llm: LLM, config: Optional[PipelineConfig] = None) -> None:
        self.store = store
        self.llm = llm
        self.cfg = config or PipelineConfig()
        self._turn = 0
        self._consent: Dict[str, int] = {}  # tag -> turn at which the window closes
        self._lock = threading.Lock()
        # the subject marker is a per-store constant: the store's record wins over the config
        stored = store._meta("subject")
        if stored and self.cfg.subject and self.cfg.subject != stored and not self.cfg.migrate_subject:
            # a second pipeline on the same store (another counterpart, a tool) must never rewrite the file:
            # it keeps the store's word for rendering and leaves the config's as an attribute only
            self.cfg.subject = stored
        if stored and self.cfg.subject and self.cfg.subject != stored:
            # the Producer changed the marker in Settings: migrate the store (deliberate, logged)
            n = store.migrate_subject(stored, self.cfg.subject)
            logging.getLogger("keepsakedb").info("subject %r -> %r (%d memories rewritten)", stored, self.cfg.subject, n)
            stored = self.cfg.subject
        self.subject = stored or self.cfg.subject
        if self.subject and not stored:
            with store._lock, store.conn:
                store._set_meta("subject", self.subject)

    # ------------------------------------------------------------- read side
    def turn(self) -> int:
        with self._lock:
            self._turn += 1
            return self._turn

    def grant_consent(self, tag: str, turns: Optional[int] = None) -> None:
        """The user agreed to talk about an archived group: its contents are injectable for
        ``turns`` turns (default config.consent_turns); session end closes every window."""
        with self._lock:
            self._consent[tag] = self._turn + (turns or self.cfg.consent_turns)

    def consented_tags(self) -> List[str]:
        return [t for t, until in self._consent.items() if until > self._turn]

    def end_session(self) -> None:
        with self._lock:
            self._consent.clear()

    def context_for(self, user_text: str, k: Optional[int] = None, title: str = "Memory") -> str:
        """The injection block: long-term facts (trust labels, archived meta-info) and, when the
        short layer is on, dated notes — both found by one embedding + one concurrent search."""
        k = k or self.cfg.context_k
        render = self.cfg.render_subject
        show = (lambda t: t.replace(self.subject, render)) if (self.subject and render and render != self.subject) else (lambda t: t)
        layers: Dict[str, int] = {"long": k}
        if self.cfg.short_layer and self.cfg.short_k > 0:
            layers["short"] = self.cfg.short_k
        found = self.store.search_layers(user_text, layers=layers, archived="include", _log=False)
        hits = found.get("long", [])
        open_tags = set(self.consented_tags())
        lines = []
        injected = set()
        for m in hits:
            if m.superseded or m.held:
                continue  # a corrected fact is never injected, consent or not; a held one is kept but not shown
            if m.archived and not (set(m.tags) & open_tags):
                topic = "/".join(m.tags) or "untagged"
                reason = f" — {m.archive_reason}" if m.archive_reason else ""
                lines.append(f"- [archived: {topic}{reason} — contents withheld; offer to discuss]")
                injected.add(m.id)
                continue
            if m.dense is not None and m.dense < self.store.drop and m.bm25 is None:
                continue
            tag_s = f" (tags: {', '.join(m.tags)})" if m.tags else ""
            pre = str(m.meta.get("prefix") or "")   # e.g. 「（舊帳 2026/10/13）」 on a forgiven hurt
            lines.append(f"- {'[high] ' if m.label == 'high' else ''}{pre}{show(m.text)}{tag_s}")
            injected.add(m.id)
        notes = []
        for m in found.get("short", []):
            if m.archived or m.held or (m.dense is not None and m.dense < self.store.drop and m.bm25 is None):
                continue
            when = time.strftime("%m/%d", time.localtime(m.created_at))
            where = m.meta.get("surface")
            notes.append(f"- [{when}{' ' + where if where else ''}] {show(m.text)}")
            injected.add(m.id)
        if self.store.log_searches:  # the calibration data: every candidate, and which ones were injected
            self.store._log_search(user_text, "context", hits, injected)
            if found.get("short"):
                self.store._log_search(user_text, "context:short", found["short"], injected)
        if not lines and not notes:
            return ""
        if lines and notes:
            return f"{title}:\nFacts (long-term):\n" + "\n".join(lines) + "\nNotes (recent, dated):\n" + "\n".join(notes)
        return f"{title}:\n" + "\n".join(lines or notes)

    # ------------------------------------------------------------ write side
    def observe(self, user_text: str, assistant_text: Optional[str] = None, *,
                turn_id: Optional[str] = None, source: Optional[str] = None) -> ObserveResult:
        """turn_id / source (who told her) are stamped into the meta of every memory this turn
        writes, so a later process (e.g. an emotion engine) can annotate them by turn."""
        if self.cfg.prefilter is not None and not self.cfg.prefilter(user_text):
            return ObserveResult()
        t0 = time.perf_counter()
        stamp = {k: v for k, v in (("turn_id", turn_id), ("source", source)) if v}
        with self._lock:  # one observe at a time (the LLM call is outside the store lock)
            return self._observe(user_text, assistant_text, t0, stamp)

    def _observe(self, user_text: str, assistant_text: Optional[str], t0: float,
                 stamp: Optional[Dict[str, Any]] = None) -> ObserveResult:
        stamp = stamp or {}
        candidates = self.store.search(user_text, k=self.cfg.max_candidates, archived="include", layer="long", _log=False)
        name_hint = (prompts.NAME_HINT.format(user_name=self.cfg.user_name, subject=self.subject or "the user")
                     if self.cfg.user_name else "")
        subject_rule = prompts.SUBJECT_RULE.format(subject=self.subject, name_hint=name_hint) if self.subject else ""
        system = prompts.SYSTEM.format(language=self.cfg.language, subject_rule=subject_rule,
                                       note_rule=prompts.NOTE_RULE.format(language=self.cfg.language) if self.cfg.short_layer else "",
                                       note_field=', "note": "one line" | null' if self.cfg.short_layer else "")
        user = prompts.build_user(user_text, assistant_text if self.cfg.user_facts_only else None,
                                  candidates, self.store.tags(include_archived=True))
        res = ObserveResult()
        cand_ids = {m.id for m in candidates}
        ops: list = []
        note: Any = None
        for attempt in range(1 + max(0, self.cfg.retries)):
            try:
                res.raw = self.llm.complete(system, user)
                data = extract_json(res.raw)
                ops = data.get("ops", [])
                note = data.get("note")
                res.error = None
                break
            except Exception as e:  # LLM or JSON failure never breaks the turn; retry once for glitches
                res.error = f"{type(e).__name__}: {e}" + (f" (after {attempt + 1} attempts)" if attempt else "")
                ops = []
                if attempt < self.cfg.retries:
                    time.sleep(self.cfg.retry_delay_s)
        for op in ops if isinstance(ops, list) else []:
            if not isinstance(op, dict):
                continue
            try:
                self._apply(op, cand_ids, res, stamp)
            except Exception as e:
                res.skipped.append({**op, "why": f"{type(e).__name__}: {e}"})
        if self.cfg.short_layer and isinstance(note, str) and note.strip():
            try:  # the episode goes to the short layer: no tags, dated by created_at, surface in meta
                meta = {"kind": "note", **stamp}
                if self.cfg.surface:
                    meta["surface"] = self.cfg.surface
                nid = self.store.add(note.strip(), [], meta, layer="short")
                res.note = note.strip()
                res.applied.append({"op": "note", "id": nid, "text": res.note})
                if self.cfg.short_cap > 0:
                    self.store.prune_layer("short", self.cfg.short_cap)
            except Exception as e:
                res.skipped.append({"op": "note", "why": f"{type(e).__name__}: {e}"})
        ms = (time.perf_counter() - t0) * 1000
        # pipeline log: candidates shown, the raw reply, what was applied/skipped — not the prompt
        self.store.log_pipeline_run(user_text, sorted(cand_ids), res.raw, res.applied, res.skipped, res.error,
                                    getattr(self.llm, "name", None), ms)
        return res

    def _apply(self, op: Dict[str, Any], cand_ids: set, res: ObserveResult,
               stamp: Optional[Dict[str, Any]] = None) -> None:
        kind = op.get("op")
        tags = [s for s in (_slug(t) for t in op.get("tags", []) or []) if s]
        text = (op.get("text") or "").strip()
        mid = op.get("id")
        reason = (op.get("reason") or "").strip()
        if kind == "noop":
            return
        if kind == "add":
            if not text:
                raise ValueError("empty text")
            new_id = self.store.add(text, tags, dict(stamp) if stamp else None)
            res.applied.append({"op": "add", "id": new_id, "text": text, "tags": tags})
        elif kind in ("update", "supersede"):
            if mid not in cand_ids:
                raise ValueError(f"id {mid} not among candidates")
            if not text:
                raise ValueError("empty text")
            if kind == "update":
                self.store.update(int(mid), text=text, tags=tags or None, reason=reason or "update")
                res.applied.append({"op": "update", "id": int(mid), "text": text, "reason": reason})
            else:
                new_id = self.store.supersede(int(mid), text, reason=reason or "superseded", tags=tags or None)
                res.applied.append({"op": "supersede", "id": int(mid), "new_id": new_id, "text": text, "reason": reason})
        elif kind in ("archive", "unarchive"):
            tag = _slug(op.get("tag", "")) if op.get("tag") else None
            if tag is None and mid not in cand_ids:
                raise ValueError("archive needs a tag or a candidate id")
            fn = self.store.archive if kind == "archive" else self.store.unarchive
            n = fn(ids=[int(mid)] if mid is not None and tag is None else None, tag=tag, reason=reason)
            res.applied.append({"op": kind, "tag": tag, "id": mid, "reason": reason, "count": n})
        elif kind == "consent":
            tag = _slug(op.get("tag", ""))
            if not tag:
                raise ValueError("consent needs a tag")
            self.grant_consent(tag)  # opens the window from the next turn (observe runs after the reply)
            res.applied.append({"op": "consent", "tag": tag, "turns": self.cfg.consent_turns})
        else:
            raise ValueError(f"unknown op {kind!r}")


__all__ = ["Pipeline", "PipelineConfig", "ObserveResult", "Memory"]
