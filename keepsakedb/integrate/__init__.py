"""L3 — wiring L1 + L2 into an agent's turn loop. Framework-agnostic.

Every agent loop has two hooks, whatever it calls them:

    before the model runs   ->  block = memory.before_turn(user_text)   # inject into the prompt
    after the reply exists  ->  memory.after_turn(user_text, reply)    # observe (background by default)
    session ends            ->  memory.end_session()                   # consent windows close

Mapping to frameworks (see docs/DESIGN.md): Hermes MemoryProvider.prefetch / sync_turn /
on_session_end; OpenClaw plugin hooks; a LangGraph node before/after the model node; or a
plain while-loop. Use ``AgentMemory`` or copy its 40 lines.
"""
from __future__ import annotations

import queue
import threading
from typing import Optional

from ..pipeline import ObserveResult, Pipeline, PipelineConfig
from ..store import Keepsake


class AgentMemory:
    def __init__(self, store: Keepsake, llm=None, config: Optional[PipelineConfig] = None,
                 background: bool = True) -> None:
        self.store = store
        self.pipeline = Pipeline(store, llm, config) if llm is not None else None
        self.background = background and self.pipeline is not None
        self._q: "queue.Queue[tuple]" = queue.Queue()
        self.last: Optional[ObserveResult] = None
        if self.background:
            self._worker = threading.Thread(target=self._run, daemon=True)
            self._worker.start()

    # ---- hooks -----------------------------------------------------------
    def before_turn(self, user_text: str, k: int = 5) -> str:
        if self.pipeline is not None:
            self.pipeline.turn()
            return self.pipeline.context_for(user_text, k)
        return self.store.context(user_text, k)  # L1 alone: trust labels + archived meta-info

    def after_turn(self, user_text: str, assistant_text: Optional[str] = None) -> Optional[ObserveResult]:
        if self.pipeline is None:
            return None
        if self.background:
            self._q.put((user_text, assistant_text))
            return None
        self.last = self.pipeline.observe(user_text, assistant_text)
        return self.last

    def end_session(self) -> None:
        if self.pipeline is not None:
            self.pipeline.end_session()

    def flush(self, timeout: Optional[float] = None) -> None:
        """Wait for queued observations (tests, shutdown)."""
        if self.background:
            self._q.join()

    # ---- worker ------------------------------------------------------------
    def _run(self) -> None:
        while True:
            user_text, assistant_text = self._q.get()
            try:
                self.last = self.pipeline.observe(user_text, assistant_text)
            finally:
                self._q.task_done()


__all__ = ["AgentMemory"]
