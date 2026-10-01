"""Operational log (the fourth stream): open/close, reindex, backups, commands, errors,
slow queries — through the standard ``logging`` module under the ``keepsakedb`` logger, so
the host decides where it goes. ``configure_logging`` attaches a JSON-lines file handler
that a daemon (556d) can tail; without it, KeepsakeDB logs nothing unless the host set up
logging itself. Zero cost when no handler is attached.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import time
from typing import Optional


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        d = {"t": round(record.created, 3), "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
             "level": record.levelname, "msg": record.getMessage()}
        if record.exc_info:
            d["exc"] = self.formatException(record.exc_info)
        return json.dumps(d, ensure_ascii=False)


def configure_logging(path: Optional[str] = None, level: int = logging.INFO, json_lines: bool = True,
                      max_bytes: int = 5_000_000, backups: int = 3) -> logging.Logger:
    """Attach a rotating file handler (JSON lines by default) or, with path=None, stderr."""
    logger = logging.getLogger("keepsakedb")
    logger.setLevel(level)
    if path:
        h: logging.Handler = logging.handlers.RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backups,
                                                                  encoding="utf-8")
    else:
        h = logging.StreamHandler()
    h.setFormatter(JsonLineFormatter() if json_lines else logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(h)
    return logger
