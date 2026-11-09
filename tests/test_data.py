"""Data: documents, gold mapping, direction, conversion, verification, splits."""
import pytest

from rag_hotpotqa.data.documents import build_question, render_text
from rag_hotpotqa.data.loader import parquet_row_to_canonical
from rag_hotpotqa.data.splits import ids_hash, make_splits, validate_splits
from rag_hotpotqa.data.verify import compare_records
from rag_hotpotqa.data.access import SplitPolicyError, check_split_allowed

from conftest import make_record


def test_stable_doc_ids_and_text(record):
    q = build_question(record)
    assert [d.doc_id for d in q.docs] == ["q1#00", "q1#01", "q1#02", "q1#03"]
    assert q.docs[0].text == "Alpha\nAlpha was born in Beta town. It is old."
    # Rebuilding gives identical ids/text/content keys (reproducible representation).
    q2 = build_question(record)
    assert [(d.doc_id, d.text, d.content_key) for d in q.docs] == [(d.doc_id, d.text, d.content_key) for d in q2.docs]


def test_gold_pair_named_by_context_index_not_answer():
    q = build_question(make_record())
    assert q.gold_pair() == (0, 1)
    assert q.oracle_directions() == [("AtoB", 0, 1), ("BtoA", 1, 0)]
    # Answer location and supporting-fact order do not change the naming.
    q2 = build_question(make_record(answer="Alpha", sf=[["Beta town", 0], ["Alpha", 0]]))
    assert q2.gold_pair() == (0, 1)
    ctx = [["Gamma", ["g"]], ["Beta town", ["Paris"]], ["Delta", ["d"]], ["Alpha", ["a Beta town"]]]
    assert build_question(make_record(context=ctx)).gold_pair() == (1, 3)


def test_comparison_uses_same_naming():
    q = build_question(make_record(qtype="comparison", answer="yes"))
    assert q.gold_pair() == (0, 1)


def test_empty_question_and_empty_doc():
    rec = make_record(question="  ", context=[["Alpha", ["x Paris"]], ["Beta town", []], ["Gamma", ["g"]]])
    q = build_question(rec)
    assert q.flags["empty_question"] is True
    assert q.docs[1].body == ""
    assert q.flags["empty_docs"] == [1]
    assert render_text("Beta town", []) == "Beta town"


def test_duplicate_title_makes_gold_ambiguous():
    ctx = [["Alpha", ["a"]], ["Alpha", ["a2"]], ["Beta town", ["Paris"]]]
    q = build_question(make_record(context=ctx))
    assert q.ambiguous_gold_titles == ["Alpha"]
    assert q.flags["duplicate_titles"] == ["Alpha"]
    assert not q.is_two_gold


def test_missing_gold_title():
    q = build_question(make_record(sf=[["Alpha", 0], ["Nowhere", 0]]))
    assert q.missing_gold_titles == ["Nowhere"] and not q.is_two_gold


def test_three_and_one_gold_docs_not_two_gold():
    three = build_question(make_record(sf=[["Alpha", 0], ["Beta town", 0], ["Gamma", 0]]))
    one = build_question(make_record(sf=[["Alpha", 0], ["Alpha", 1]]))
    assert three.n_gold == 3 and not three.is_two_gold
    assert one.n_gold == 1 and not one.is_two_gold


def test_sf_sentence_out_of_range_flagged():
    q = build_question(make_record(sf=[["Alpha", 9], ["Beta town", 0]]))
    assert q.flags["sf_sentence_out_of_range"] == [["Alpha", 9]]
    assert q.is_two_gold  # doc-level gold unaffected


def test_parquet_row_conversion():
    row = {"id": "x", "question": "q?", "answer": "a", "type": "bridge", "level": "hard",
           "supporting_facts": {"title": ["T1", "T2"], "sent_id": [0, 2]},
           "context": {"title": ["T1", "T2"], "sentences": [["s1"], ["s2", " s3", " s4"]]}}
    rec = parquet_row_to_canonical(row)
    assert rec["supporting_facts"] == [["T1", 0], ["T2", 2]]
    assert rec["context"] == [["T1", ["s1"]], ["T2", ["s2", " s3", " s4"]]]
    with pytest.raises(ValueError):
        parquet_row_to_canonical({**row, "context": {"title": ["T1"], "sentences": []}})


def test_compare_records_detects_semantic_differences(record):
    same = compare_records([record], [dict(record)])
    assert same["semantically_identical"]
    changed = dict(record, context=[[t, list(s)] for t, s in record["context"]])
    changed["context"][1][1][0] = "Beta town is in Lyon."
    diff = compare_records([record], [changed])
    assert not diff["semantically_identical"] and diff["field_mismatch_counts"] == {"context": 1}
    swapped = dict(record, context=list(reversed(record["context"])))
    assert compare_records([record], [swapped])["field_mismatch_counts"] == {"context": 1}
    sf_swapped = dict(record, supporting_facts=list(reversed(record["supporting_facts"])))
    assert compare_records([record], [sf_swapped])["field_mismatch_counts"] == {"supporting_facts": 1}
    missing = compare_records([record], [])
    assert missing["ids_only_in_reference"] == 1 and not missing["semantically_identical"]


def _pool(n=1000):
    recs = []
    for i in range(n):
        recs.append({"_id": f"id{i:05d}", "type": "comparison" if i % 5 == 0 else "bridge",
                     "level": ["easy", "medium", "hard"][i % 3]})
    return recs


def test_splits_disjoint_exact_stratified_deterministic():
    recs = _pool()
    sizes = {"exploration": 100, "training": 300, "validation": 100}
    s1 = make_splits(recs, sizes, seed=42)
    s2 = make_splits(list(reversed(recs)), sizes, seed=42)
    assert s1 == s2  # independent of row order
    validate_splits(s1, sizes)
    for ids in s1.values():
        frac = sum(i in {r["_id"] for r in recs if r["type"] == "comparison"} for i in ids) / len(ids)
        assert abs(frac - 0.2) < 0.02
    s3 = make_splits(recs, sizes, seed=7)
    assert ids_hash(s3["exploration"]) != ids_hash(s1["exploration"])


def test_split_duplicate_ids_rejected():
    recs = _pool(10) + [{"_id": "id00001", "type": "bridge", "level": "easy"}]
    with pytest.raises(ValueError, match="Duplicate"):
        make_splits(recs, {"exploration": 2}, seed=42)
    with pytest.raises(ValueError, match="duplicate"):
        validate_splits({"exploration": ["a", "a"]}, {"exploration": 2})


def test_dev_or_validation_leak_into_training_detected():
    splits = {"training": ["a", "b"], "validation": ["c", "d"]}
    with pytest.raises(ValueError, match="official_dev"):
        validate_splits(splits, {"training": 2, "validation": 2}, forbidden={"official_dev": {"b"}})
    with pytest.raises(ValueError, match="overlap"):
        validate_splits({"training": ["a", "b"], "validation": ["b", "c"]}, {"training": 2, "validation": 2})


def test_split_policy_by_purpose():
    check_split_allowed("exploration", "oer")
    check_split_allowed("training", "train")
    check_split_allowed("validation", "select")
    check_split_allowed("official_dev", "final")
    for split, purpose in [("validation", "oer"), ("training", "oer"), ("official_dev", "oer"),
                           ("exploration", "train"), ("validation", "train"), ("official_dev", "train"),
                           ("training", "select"), ("exploration", "final")]:
        with pytest.raises(SplitPolicyError):
            check_split_allowed(split, purpose)
    with pytest.raises(SplitPolicyError):
        check_split_allowed("exploration", "anything")
