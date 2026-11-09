"""Authorisation gate of the final-evaluation datasets: V-report / B-C holdout are never readable without an approved,
FROZEN authorisation whose freeze manifest matches its sha256 (only the refusal paths are exercised here)."""
from __future__ import annotations

import json

import pytest

from rag_hotpotqa.data.access import SplitPolicyError
from rag_hotpotqa.data.eval_parts import check_authorisation
from rag_hotpotqa.utils.config import load_settings


@pytest.mark.parametrize("part", ["v_report", "bc_holdout"])
def test_refused_without_authorisation(part):
    with pytest.raises(SplitPolicyError):
        check_authorisation(load_settings(), part, None)


def test_refused_when_not_frozen_or_not_approved(tmp_path):
    st = load_settings()
    a = tmp_path / "auth.json"
    a.write_text(json.dumps({"status": "DRAFT", "approved_parts": ["v_report"]}))
    with pytest.raises(SplitPolicyError, match="does not approve"):
        check_authorisation(st, "v_report", str(a))
    a.write_text(json.dumps({"status": "FROZEN", "approved_parts": ["bc_holdout"]}))
    with pytest.raises(SplitPolicyError, match="does not approve"):
        check_authorisation(st, "v_report", str(a))


def test_refused_when_freeze_manifest_sha_mismatch(tmp_path):
    st = load_settings()
    fm = tmp_path / "freeze.json"
    fm.write_text("{}")
    a = tmp_path / "auth.json"
    a.write_text(json.dumps({"status": "FROZEN", "approved_parts": ["v_report"], "freeze_manifest": str(fm), "freeze_manifest_sha256": "0" * 64}))
    with pytest.raises(SplitPolicyError, match="freeze manifest"):
        check_authorisation(st, "v_report", str(a))


def test_v_select_needs_no_authorisation():
    assert check_authorisation(load_settings(), "v_select", None) is None
