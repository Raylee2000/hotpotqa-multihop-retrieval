"""Crash-safe IO helpers: atomic writes, hashing, resumable JSONL."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import safe_path

HASH_CHUNK = 1 << 20


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


REPLACE_RETRIES = 50
REPLACE_BACKOFF_S = 0.02


def _replace_with_retry(src: str, dst: Path) -> None:
    """os.replace, retrying on Windows sharing violations (another process is
    replacing or reading ``dst`` at the same instant)."""
    for attempt in range(REPLACE_RETRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == REPLACE_RETRIES - 1:
                raise
            time.sleep(REPLACE_BACKOFF_S * (1 + attempt % 5))


def atomic_write_text(path: Path | str, text: str, root: Path) -> Path:
    """Write via temp file + os.replace so readers never see a partial file."""
    target = safe_path(path, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=target.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        _replace_with_retry(tmp, target)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return target


def atomic_write_json(path: Path | str, obj: Any, root: Path) -> Path:
    return atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", root)


def read_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    """Yield records; a truncated final line (crash mid-write) is skipped, any
    other corrupt line raises."""
    p = Path(path)
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1 and not line.endswith("\n"):
                return
            raise


class ResumableJsonl:
    """Append-only result log keyed by ``key_field``.

    On open, existing complete records are loaded so already-finished keys are
    skipped (resume) and never written twice (dedup). A trailing partial line
    from a crash is truncated away before appending.
    """

    def __init__(self, path: Path | str, root: Path, key_field: str = "qid", keep_records: bool = True):
        self.path = safe_path(path, root)
        self.key_field = key_field
        self.keep_records = keep_records
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done: dict[str, dict[str, Any] | None] = {}
        for rec in read_jsonl(self.path):
            self.done[str(rec[key_field])] = rec if keep_records else None
        self._repair_tail()

    def _repair_tail(self) -> None:
        if not self.path.exists():
            return
        data = self.path.read_bytes()
        if data and not data.endswith(b"\n"):
            cut = data.rfind(b"\n") + 1
            with open(self.path, "r+b") as fh:
                fh.truncate(cut)

    def __contains__(self, key: str) -> bool:
        return str(key) in self.done

    def append(self, record: dict[str, Any]) -> bool:
        key = str(record[self.key_field])
        if key in self.done:
            return False
        with open(self.path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self.done[key] = record if self.keep_records else None
        return True

    def extend(self, records: Iterable[dict[str, Any]]) -> int:
        return sum(self.append(r) for r in records)
