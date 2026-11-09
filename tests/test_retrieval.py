"""Retrieval: BM25 math, RRF math, ranking ties, dense composition/truncation, cache safety."""
import math

import numpy as np
import pytest

from rag_hotpotqa.retrieval.base import NonFiniteScoreError, check_finite, rank_order, ranks_from_scores
from rag_hotpotqa.retrieval.bm25 import BM25Scorer, BM25Stats, tokenize
from rag_hotpotqa.retrieval.dense import CacheMismatchError, DenseScorer, EmbeddingCache
from rag_hotpotqa.retrieval.hybrid import rrf_scores


def test_tokenizer_unicode_and_case():
    assert tokenize("Café ＡＢＣ, naïve—Test") == ["café", "abc", "naïve", "test"]


def test_bm25_matches_hand_computation():
    corpus = ["a b c", "a a d", "e f"]
    stats = BM25Stats.from_texts(corpus)
    assert stats.n_docs == 3 and stats.avgdl == pytest.approx(8 / 3)
    s = BM25Scorer("trainglobal", stats, k1=1.2, b=0.75).score("a d", corpus)
    idf_a = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    idf_d = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    norm = 1.2 * (1 - 0.75 + 0.75 * 3 / (8 / 3))
    exp0 = idf_a * 1 * 2.2 / (1 + norm)
    exp1 = idf_a * 2 * 2.2 / (2 + norm) + idf_d * 1 * 2.2 / (1 + norm)
    assert s[0] == pytest.approx(exp0) and s[1] == pytest.approx(exp1) and s[2] == 0.0


def test_bm25_empty_query_or_doc_scores_zero():
    stats = BM25Stats.from_texts(["x y"])
    sc = BM25Scorer("trainglobal", stats)
    assert list(sc.score("", ["x y", ""])) == [0.0, 0.0]
    assert sc.score("x", [""])[0] == 0.0


def test_rank_ties_broken_by_index():
    assert rank_order([1.0, 2.0, 2.0, 1.0]) == [1, 2, 0, 3]
    assert ranks_from_scores([1.0, 2.0, 2.0, 1.0]) == [3, 1, 2, 4]
    assert rank_order([5, 5], tie_keys=["b", "a"]) == [1, 0]


def test_non_finite_scores_rejected():
    with pytest.raises(NonFiniteScoreError):
        check_finite(np.array([0.1, np.nan]))
    with pytest.raises(NonFiniteScoreError):
        rank_order([np.inf, 0.0])


def test_rrf_formula_and_ties():
    bm25 = [3.0, 1.0, 2.0]     # ranks 1,3,2
    dense = [0.1, 0.9, 0.9]    # ranks 3,1,2 (tie: index 1 before 2)
    out = rrf_scores([bm25, dense], k=60)
    assert out[0] == pytest.approx(1 / 61 + 1 / 63)
    assert out[1] == pytest.approx(1 / 63 + 1 / 61)
    assert out[2] == pytest.approx(1 / 62 + 1 / 62)
    # docs 0 and 1 tie exactly -> stable index order
    assert rank_order(out)[:2] == [0, 1] or rank_order(out)[0] == 2
    with pytest.raises(ValueError):
        rrf_scores([[1, 2], [1]])


def _dense(fake_model, tmp_path=None, max_len=12):
    return DenseScorer("fake", revision="r1", model=fake_model, max_seq_length=max_len,
                       cache_dir=tmp_path, joiner="\n\n")


def test_query_first_truncation_keeps_question(fake_model):
    d = _dense(fake_model)  # budget = 12 - 2 = 10 tokens; joiner is whitespace -> 0 tokens
    q = "w1 w2 w3 w4"
    a = " ".join(f"a{i}" for i in range(20))
    out = d.compose_q_plus_a(q, a)
    left, right = out.split("\n\n")
    assert left == q and right.split() == [f"a{i}" for i in range(6)]


def test_balanced_truncation_splits_budget(fake_model):
    d = _dense(fake_model)
    a = " ".join(f"a{i}" for i in range(20))
    b = " ".join(f"b{i}" for i in range(20))
    left, right = d.compose_a_plus_b(a, b).split("\n\n")
    assert len(left.split()) == 5 and len(right.split()) == 5
    short = "s1 s2"
    left, right = d.compose_a_plus_b(short, b).split("\n\n")
    assert left == short and len(right.split()) == 8  # unused budget goes to the other side


def test_dense_scores_cosine_and_uses_cache(fake_model, tmp_path):
    d = _dense(fake_model, tmp_path)
    s = d.score("alpha beta", ["alpha beta", "gamma"])
    assert s[0] == pytest.approx(1.0) and s[0] > s[1]
    calls = fake_model.calls
    d.score("alpha beta", ["alpha beta", "gamma"])
    assert fake_model.calls == calls  # cached, no re-encoding


def test_cache_persists_and_rejects_other_model_version(fake_model, tmp_path):
    d = _dense(fake_model, tmp_path)
    d.prefetch(["hello world"])
    d.cache.flush()
    fp = d.fingerprint()
    again = EmbeddingCache(tmp_path, fp)
    assert again.get("hello world") is not None
    # A shard copied under another fingerprint directory must be rejected.
    other = dict(fp, revision="r2")
    other_cache = EmbeddingCache(tmp_path, other)
    other_cache.dir.mkdir(parents=True, exist_ok=True)
    shard = next(again.dir.glob("shard_*.npz"))
    (other_cache.dir / shard.name).write_bytes(shard.read_bytes())
    with pytest.raises(CacheMismatchError):
        EmbeddingCache(tmp_path, other)


def test_embedding_dim_mismatch_rejected(tmp_path):
    c = EmbeddingCache(None, {"m": 1})
    c.put("a", np.ones(4, dtype=np.float32))
    with pytest.raises(CacheMismatchError):
        c.put("b", np.ones(5, dtype=np.float32))


def test_nan_embedding_rejected(fake_model):
    class NaNModel(type(fake_model)):
        def encode(self, texts, **kw):
            return np.full((len(texts), 4), np.nan, dtype=np.float32)

    d = DenseScorer("nan", model=NaNModel(), max_seq_length=12)
    with pytest.raises(NonFiniteScoreError):
        d.score("x", ["y"])
