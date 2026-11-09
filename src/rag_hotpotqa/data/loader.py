"""Load HotpotQA into the *original JSON schema* regardless of source format.

Canonical record (identical to hotpot_*_v1*.json):
    {"_id", "question", "answer", "type", "level",
     "supporting_facts": [[title, sent_id], ...],
     "context": [[title, [sentence, ...]], ...]}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

CANONICAL_FIELDS = ("_id", "question", "answer", "type", "level", "supporting_facts", "context")


def parquet_row_to_canonical(row: dict[str, Any]) -> dict[str, Any]:
    """Convert one HF parquet row (columnar structs) to the original JSON schema."""
    sf = row["supporting_facts"]
    ctx = row["context"]
    if len(sf["title"]) != len(sf["sent_id"]):
        raise ValueError(f"{row['id']}: supporting_facts title/sent_id length mismatch")
    if len(ctx["title"]) != len(ctx["sentences"]):
        raise ValueError(f"{row['id']}: context title/sentences length mismatch")
    return {
        "_id": row["id"],
        "question": row["question"],
        "answer": row["answer"],
        "type": row["type"],
        "level": row["level"],
        "supporting_facts": [[t, int(i)] for t, i in zip(sf["title"], sf["sent_id"])],
        "context": [[t, list(s)] for t, s in zip(ctx["title"], ctx["sentences"])],
    }


def iter_parquet(paths: Iterable[Path | str], batch_size: int = 2000) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                yield parquet_row_to_canonical(row)


def load_original_json(path: Path | str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return [{k: rec[k] for k in CANONICAL_FIELDS} for rec in data]


def write_canonical_jsonl(records: Iterable[dict[str, Any]], path: Path, root: Path) -> int:
    from ..utils.io import atomic_write_text

    lines = [json.dumps(r, ensure_ascii=False, sort_keys=True) for r in records]
    atomic_write_text(path, "\n".join(lines) + "\n", root)
    return len(lines)


def load_canonical_jsonl(path: Path | str, ids: set[str] | None = None) -> list[dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if ids is None or rec["_id"] in ids:
                out.append(rec)
    return out
