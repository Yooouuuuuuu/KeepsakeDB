"""L2 — the memory pipeline. Pluggable: nothing in L1 depends on it.

observe(turn)      one LLM pass decides add / update / supersede / archive / noop, with the
                   most similar memories and the tag groups as context (mem0's ADD/UPDATE/DELETE
                   pattern + the C2 tag-organized, conflict-aware merge).
context_for(text)  the injection block; archived groups show as meta-info only, unless the
                   user granted consent (a temp window of N turns, capped at session end).
"""
from __future__ import annotations

import re
import threading
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


@dataclass
class ObserveResult:
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

    def context_for(self, user_text: str, k: Optional[int] = None, title: str = "Long-term memory") -> str:
        k = k or self.cfg.context_k
        hits = self.store.search(user_text, k=k, archived="include")
        open_tags = set(self.consented_tags())
        lines = []
        for m in hits:
            if m.superseded:
                continue  # a corrected fact is never injected, consent or not
            if m.archived and not (set(m.tags) & open_tags):
                topic = "/".join(m.tags) or "untagged"
                reason = f" — {m.archive_reason}" if m.archive_reason else ""
                lines.append(f"- [archived: {topic}{reason} — contents withheld; offer to discuss]")
                continue
            if m.dense is not None and m.dense < self.store.drop and m.bm25 is None:
                continue
            tag_s = f" (tags: {', '.join(m.tags)})" if m.tags else ""
            lines.append(f"- {'[high] ' if m.label == 'high' else ''}{m.text}{tag_s}")
        return f"{title}:\n" + "\n".join(lines) if lines else ""

    # ------------------------------------------------------------ write side
    def observe(self, user_text: str, assistant_text: Optional[str] = None) -> ObserveResult:
        if self.cfg.prefilter is not None and not self.cfg.prefilter(user_text):
            return ObserveResult()
        candidates = self.store.search(user_text, k=self.cfg.max_candidates, archived="include")
        system = prompts.SYSTEM.format(language=self.cfg.language)
        user = prompts.build_user(user_text, assistant_text if self.cfg.user_facts_only else None,
                                  candidates, self.store.tags(include_archived=True))
        res = ObserveResult()
        try:
            res.raw = self.llm.complete(system, user)
            ops = extract_json(res.raw).get("ops", [])
        except Exception as e:  # LLM or JSON failure never breaks the turn
            res.error = f"{type(e).__name__}: {e}"
            return res
        cand_ids = {m.id for m in candidates}
        for op in ops if isinstance(ops, list) else []:
            if not isinstance(op, dict):
                continue
            try:
                self._apply(op, cand_ids, res)
            except Exception as e:
                res.skipped.append({**op, "why": f"{type(e).__name__}: {e}"})
        return res

    def _apply(self, op: Dict[str, Any], cand_ids: set, res: ObserveResult) -> None:
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
            new_id = self.store.add(text, tags)
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
        else:
            raise ValueError(f"unknown op {kind!r}")


__all__ = ["Pipeline", "PipelineConfig", "ObserveResult", "Memory"]
