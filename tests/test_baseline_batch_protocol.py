"""Guard the cache-prefill batching that caused the V-select regression."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from conftest import FakeModel, make_record
from rag_hotpotqa.data.documents import build_question
from rag_hotpotqa.experiments.runner import _prefetch
from rag_hotpotqa.retrieval.dense import DenseScorer
from rag_hotpotqa.retrieval.similarity import Similarity

spec = importlib.util.spec_from_file_location("final_baselines", Path(__file__).resolve().parents[1] / "scripts/final_eval/run_baselines.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class RecordingModel(FakeModel):
    def __init__(self):
        super().__init__()
        self.batch_inputs = []

    def encode(self, texts, **kwargs):
        self.batch_inputs.append(list(texts))
        return super().encode(texts, **kwargs)


def setup_protocol():
    qs = [build_question(make_record(qid=f"q{i}", question=f"Question {i}",
                context=[[f"Title {i} {j}", [f"Body {i} {j}"]] for j in range(4)],
                sf=[[f"Title {i} 0", 0], [f"Title {i} 1", 0]])) for i in range(32)]
    models = {n: RecordingModel() for n in ("dense", "general_best")}
    sims = {n: Similarity(n, [DenseScorer(n, model=m)]) for n,m in models.items()}
    return qs, models, sims


def test_prefill_replays_partial_final_chunk_without_predictions():
    qs, models, sims = setup_protocol()
    entry.prefill_embeddings(qs, sims, 20, 16)
    for n, sim in sims.items():
        _prefetch(qs[:16], sim, ["O"], {})
        _prefetch(qs[16:32], sim, ["O"], {})
        assert [len(ts) for ts in models[n].batch_inputs] == [80, 20, 60]
        assert len(sim.components[0].cache.mem) == 160


def test_zero_prefill_keeps_cold_protocol():
    qs, models, sims = setup_protocol()
    entry.prefill_embeddings(qs, sims, 0, 16)
    for n, sim in sims.items():
        assert not models[n].batch_inputs
        _prefetch(qs[:16], sim, ["O"], {})
        _prefetch(qs[16:], sim, ["O"], {})
        assert [len(ts) for ts in models[n].batch_inputs] == [80, 80]


@pytest.mark.parametrize("n", [-1, 33])
def test_invalid_prefill_fails_before_encoding(n):
    qs, models, sims = setup_protocol()
    with pytest.raises(ValueError):
        entry.prefill_embeddings(qs, sims, n, 16)
    assert all(not m.batch_inputs for m in models.values())
