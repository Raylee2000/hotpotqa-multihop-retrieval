"""Shared fixtures: hand-built HotpotQA records and a deterministic fake encoder."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest


def make_record(qid="q1", qtype="bridge", answer="Paris", context=None, sf=None, question="Where was X born?"):
    context = context if context is not None else [
        ["Alpha", ["Alpha was born in Beta town.", " It is old."]],
        ["Beta town", ["Beta town is in Paris.", " Nice place."]],
        ["Gamma", ["Gamma is unrelated."]],
        ["Delta", ["Delta is also unrelated."]],
    ]
    sf = sf if sf is not None else [["Alpha", 0], ["Beta town", 0]]
    return {"_id": qid, "question": question, "answer": answer, "type": qtype, "level": "medium",
            "supporting_facts": sf, "context": context}


@pytest.fixture
def record():
    return make_record()


class FakeTokenizer:
    """Whitespace tokenizer with character offsets (mimics HF fast tokenizer API)."""

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        spans = [m.span() for m in re.finditer(r"\S+", text)]
        out = {"input_ids": list(range(len(spans)))}
        if return_offsets_mapping:
            out["offset_mapping"] = spans
        return out


class FakeModel:
    """Bag-of-words hashing encoder: deterministic, normalised, dim 64."""

    def __init__(self, dim=64):
        self.dim = dim
        self.tokenizer = FakeTokenizer()
        self.max_seq_length = 256
        self.calls = 0

    def encode(self, texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = re.findall(r"\w+", t.lower())[: self.max_seq_length - 2]
            for tok in toks:
                out[i, sum(map(ord, tok)) % self.dim] += 1.0
            n = np.linalg.norm(out[i])
            if normalize_embeddings and n > 0:
                out[i] /= n
        return out


@pytest.fixture
def fake_model():
    return FakeModel()


@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    return tmp_path
