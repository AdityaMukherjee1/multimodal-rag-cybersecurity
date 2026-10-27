"""In-memory thread-safe log handler for live frontend event streaming.

Captures actual application logs emitted by src.retrieval, src.multimodal,
src.embeddings, etc.
"""

from __future__ import annotations

import collections
import logging
import threading
from dataclasses import dataclass
from datetime import datetime


@dataclass
class LogEntry:
    """A single logged event."""
    timestamp: str
    level: str
    logger_name: str
    message: str


class LiveLogHandler(logging.Handler):
    """Thread-safe circular in-memory buffer of application logs."""

    def __init__(self, max_entries: int = 200) -> None:
        super().__init__()
        self.max_entries = max_entries
        self._buffer: collections.deque[LogEntry] = collections.deque(maxlen=max_entries)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            entry = LogEntry(
                timestamp=datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3],
                level=record.levelname,
                logger_name=record.name.split(".")[-1],
                message=record.getMessage(),
            )
            with self._lock:
                self._buffer.append(entry)
        except Exception:
            self.handleError(record)

    def get_entries(self, limit: int | None = None) -> list[LogEntry]:
        """Return recent log entries in chronological order."""
        with self._lock:
            entries = list(self._buffer)
        if limit:
            return entries[-limit:]
        return entries

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()


# Global singleton instance
_log_handler_instance: LiveLogHandler | None = None
_handler_lock = threading.Lock()


def get_live_log_handler() -> LiveLogHandler:
    """Retrieve or initialize the global live log handler."""
    global _log_handler_instance
    with _handler_lock:
        if _log_handler_instance is None:
            _log_handler_instance = LiveLogHandler(max_entries=300)
            _log_handler_instance.setLevel(logging.INFO)
            # Attach to root and major project loggers
            root_logger = logging.getLogger()
            root_logger.addHandler(_log_handler_instance)
            
            # Explicitly attach to src loggers
            for mod in ["src", "src.retrieval", "src.multimodal", "src.embeddings", "src.pipeline", "frontend"]:
                logging.getLogger(mod).addHandler(_log_handler_instance)
        return _log_handler_instance
