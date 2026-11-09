"""Model Artifact Contract: missing / corrupted files, feature-order mismatch, reload mismatch, completion rule."""
from __future__ import annotations

import json

import numpy as np
import pytest

from rag_hotpotqa.utils.artifacts import (ArtifactError, check_feature_schema, compare_predictions, is_complete, mark_complete,
                                          verify_manifest, write_manifest)


def _art(tmp_path):
    (tmp_path / "model.bin").write_bytes(b"weights")
    (tmp_path / "schema.json").write_text(json.dumps({"features": ["a", "b"]}))
    write_manifest(tmp_path, {"model": tmp_path / "model.bin", "schema": tmp_path / "schema.json"}, model_class="toy")
    return tmp_path


def test_valid_artifact_completes(tmp_path):
    d = _art(tmp_path)
    assert not is_complete(d)                      # a manifest alone is not completion
    mark_complete(d, {"status": "PASS"})
    assert is_complete(d)


def test_missing_model_file_fails(tmp_path):
    d = _art(tmp_path)
    (d / "model.bin").unlink()
    with pytest.raises(ArtifactError, match="missing"):
        verify_manifest(d)
    with pytest.raises(ArtifactError):
        write_manifest(d, {"model": d / "model.bin"})


def test_corrupted_model_file_fails(tmp_path):
    d = _art(tmp_path)
    mark_complete(d, {"status": "PASS"})
    (d / "model.bin").write_bytes(b"tampered")
    with pytest.raises(ArtifactError, match="corrupted"):
        verify_manifest(d)
    assert not is_complete(d)                      # completion is revoked by corruption


def test_missing_manifest_and_result_only_is_not_complete(tmp_path):
    (tmp_path / "result.json").write_text("{}")
    assert not is_complete(tmp_path)
    with pytest.raises(ArtifactError, match="missing"):
        verify_manifest(tmp_path)


def test_feature_order_mismatch_fails():
    check_feature_schema(["a", "b", "c"], ["a", "b", "c"])
    with pytest.raises(ArtifactError, match="feature order"):
        check_feature_schema(["a", "b", "c"], ["a", "c", "b"])


def test_reload_prediction_mismatch_fails(tmp_path):
    assert compare_predictions(np.array([0.1, 0.2]), np.array([0.1, 0.2 + 1e-12]), atol=1e-9) < 1e-9
    with pytest.raises(ArtifactError, match="differ"):
        compare_predictions(np.array([0.1, 0.2]), np.array([0.1, 0.3]), atol=1e-9)
    with pytest.raises(ArtifactError, match="shape"):
        compare_predictions(np.zeros(3), np.zeros(2), atol=1e-9)
    d = _art(tmp_path)
    with pytest.raises(ArtifactError, match="PASS"):
        mark_complete(d, {"status": "FAIL"})
    assert not is_complete(d)
