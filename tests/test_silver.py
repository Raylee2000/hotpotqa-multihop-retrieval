"""Silver (derived) bridge labels: rule ids, link direction, question-title fallback, entity span."""
from __future__ import annotations

from rag_hotpotqa.data.documents import build_question
from rag_hotpotqa.training.silver import core_title, derive_bridge, find_title

from conftest import make_record


def q(context, sf, question="Where was the founder of Alpha born?", answer="Paris", qtype="bridge"):
    return build_question(make_record(context=context, sf=sf, question=question, answer=answer, qtype=qtype))


def test_title_core_and_span():
    assert core_title("Paris (band)") == "Paris"
    assert find_title("Beta Town (film)", "He moved to beta town in 1990.") == (12, 21)
    assert find_title("Al", "Al is short") is None  # too short to match


def test_link_rule_gives_order_and_entity():
    ctx = [["Alpha", ["Alpha was founded by John Smith.", " It is old."]],
           ["John Smith", ["John Smith was born in Paris."]],
           ["Gamma", ["Unrelated."]]]
    s = derive_bridge(q(ctx, [["Alpha", 0], ["John Smith", 0]]))
    assert (s.first, s.second, s.hop_rule, s.link_dirs) == (0, 1, "S-HOP-LINK", 1)
    assert s.entity["text"] == "John Smith" and s.entity["sentence_id"] == 0 and s.entity["rule"] == "S-ENT-LINK"
    assert s.ans_second == 1  # answer check agrees: Paris is in B


def test_question_title_fallback_and_no_entity():
    ctx = [["Alpha", ["Alpha was founded by a man."]], ["Zed", ["Zed was born in Paris."]], ["Gamma", ["Unrelated."]]]
    s = derive_bridge(q(ctx, [["Alpha", 0], ["Zed", 0]]))
    assert (s.first, s.second, s.hop_rule, s.entity) == (0, 1, "S-HOP-QTITLE", None)


def test_undecided_and_non_bridge():
    ctx = [["Alpha", ["Alpha mentions Zed."]], ["Zed", ["Zed mentions Alpha."]], ["Gamma", ["Unrelated."]]]
    s = derive_bridge(q(ctx, [["Alpha", 0], ["Zed", 0]], question="Which is older?"))
    assert s.hop_rule is None and s.link_dirs == 2 and s.first is None
    assert derive_bridge(q(ctx, [["Alpha", 0], ["Zed", 0]], qtype="comparison")) is None
