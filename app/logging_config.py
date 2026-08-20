import json
import logging
import re
import sys
import time

_REDACT_KEYS = {"password", "db_password", "token", "secret", "authorization", "api_key"}
_CONN_STRING_RE = re.compile(r"(://[^:/@]+:)[^@]+(@)")


def redact(value):
    """Recursively strip credential-shaped fields/substrings before logging."""
    if isinstance(value, dict):
        return {
            k: ("***" if k.lower() in _REDACT_KEYS else redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return _CONN_STRING_RE.sub(r"\1***\2", value)
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        event = getattr(record, "event", None)
        if event:
            payload.update(redact(event))
        return json.dumps(payload, default=str)


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("text2sql")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    return logger


audit_logger = setup_logging()


def log_audit(**event):
    """Emit one structured audit event per SRS Section 21. Never pass raw
    credentials/tokens in `event` -- redact() strips known key names and
    connection-string password segments as a second line of defense."""
    audit_logger.info(event.get("status", "request"), extra={"event": event})


def new_request_id() -> str:
    import uuid
    return uuid.uuid4().hex


def now_ms() -> float:
    return time.perf_counter() * 1000
