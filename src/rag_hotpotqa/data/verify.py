"""Field-level equivalence check between two HotpotQA record sets aligned by _id."""
from __future__ import annotations

from collections import Counter
from typing import Any

from .loader import CANONICAL_FIELDS


def _diff_context(a: list, b: list) -> str | None:
    if len(a) != len(b):
        return f"context length {len(a)} != {len(b)}"
    for i, ((ta, sa), (tb, sb)) in enumerate(zip(a, b)):
        if ta != tb:
            return f"context[{i}] title {ta!r} != {tb!r}"
        if len(sa) != len(sb):
            return f"context[{i}] ({ta!r}) sentence count {len(sa)} != {len(sb)}"
        for j, (x, y) in enumerate(zip(sa, sb)):
            if x != y:
                return f"context[{i}] ({ta!r}) sentence {j} differs: {x[:60]!r} != {y[:60]!r}"
    return None


def compare_records(reference: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare every canonical field exactly (strings byte-for-byte after JSON
    decoding; list order significant). ``reference`` is the original JSON."""
    ref = {r["_id"]: r for r in reference}
    cand = {r["_id"]: r for r in candidate}
    dup_ref = len(reference) - len(ref)
    dup_cand = len(candidate) - len(cand)
    only_ref = sorted(set(ref) - set(cand))
    only_cand = sorted(set(cand) - set(ref))
    field_mismatch: Counter[str] = Counter()
    examples: list[dict[str, str]] = []
    order_same = [r["_id"] for r in reference] == [r["_id"] for r in candidate]
    for qid in sorted(set(ref) & set(cand)):
        a, b = ref[qid], cand[qid]
        for field in CANONICAL_FIELDS:
            if field == "context":
                msg = _diff_context(a["context"], b["context"])
            elif a[field] != b[field]:
                msg = f"{field}: {str(a[field])[:80]!r} != {str(b[field])[:80]!r}"
            else:
                msg = None
            if msg:
                field_mismatch[field] += 1
                if len(examples) < 20:
                    examples.append({"_id": qid, "detail": msg})
    identical = not (only_ref or only_cand or dup_ref or dup_cand or field_mismatch)
    return {
        "n_reference": len(reference),
        "n_candidate": len(candidate),
        "duplicate_ids_reference": dup_ref,
        "duplicate_ids_candidate": dup_cand,
        "ids_only_in_reference": len(only_ref),
        "ids_only_in_candidate": len(only_cand),
        "ids_compared": len(set(ref) & set(cand)),
        "record_order_identical": order_same,
        "fields_compared": list(CANONICAL_FIELDS),
        "field_mismatch_counts": dict(field_mismatch),
        "mismatch_examples": examples,
        "semantically_identical": identical,
    }
