"""Release-specific checks: pickle-free R5 inference, safe model extraction, integrity checks, no unsafe calls."""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pytest

from rag_hotpotqa.ranking.r5_trees import FORMAT, R5Trees
from rag_hotpotqa.release import integrity
from rag_hotpotqa.release.integrity import IntegrityError, safe_extract, sha256_bytes, text_sha256, verify_model_dir

ROOT = Path(__file__).resolve().parents[1]


def _export(lr, trees, features):
    parts = {k: [] for k in ("left", "right", "feature", "threshold", "value")}
    offsets = [0]
    for t in trees:
        tr = t.tree_
        parts["left"].append(tr.children_left.astype(np.int64))
        parts["right"].append(tr.children_right.astype(np.int64))
        parts["feature"].append(tr.feature.astype(np.int64))
        parts["threshold"].append(tr.threshold.astype(np.float64))
        parts["value"].append(tr.value[:, 0, 0].astype(np.float64))
        offsets.append(offsets[-1] + tr.node_count)
    return R5Trees(lr, features=features, offsets=np.asarray(offsets, dtype=np.int64), **{k: np.concatenate(v) for k, v in parts.items()})


def test_r5_trees_match_sklearn_exactly(tmp_path):
    from sklearn.tree import DecisionTreeRegressor

    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(400, 6)), rng.normal(size=400)
    trees = [DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=i).fit(X, y + i) for i in range(7)]
    feats = [f"f{i}" for i in range(6)]
    m = _export(0.1, trees, feats)
    Z = rng.normal(size=(300, 6)) * 3
    assert np.array_equal(m.predict(Z), 0.1 * sum(t.predict(Z) for t in trees))
    p = tmp_path / "r5.npz"
    np.savez(p, meta_json=np.asarray(json.dumps({"format": FORMAT, "lr": 0.1, "features": feats})), offsets=m.offsets,
             left=m.left, right=m.right, feature=m.feature, threshold=m.threshold, value=m.value)
    assert np.array_equal(R5Trees.load(p).predict(Z), m.predict(Z))
    with pytest.raises(ValueError):
        m.predict(np.full((2, 6), np.nan))
    with pytest.raises(ValueError):
        m.predict(Z[:, :5])


def _spec(files: dict[str, bytes], dirs=()):
    return {"files": {k: {"sha256": sha256_bytes(v), "bytes": len(v)} for k, v in files.items()}, "empty_dirs": list(dirs)}


def _zip(path, entries):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in entries.items():
            z.writestr(name, data)


def test_safe_extract_accepts_only_listed_files(tmp_path):
    files = {"a.json": b"{}", "sub/b.txt": b"x"}
    spec = _spec(files, ["2_Normalize"])
    good = tmp_path / "good.zip"
    _zip(good, {**files, "2_Normalize/": b""})
    safe_extract(good, spec, tmp_path / "out")
    assert (tmp_path / "out/sub/b.txt").read_bytes() == b"x" and (tmp_path / "out/2_Normalize").is_dir()

    extra = tmp_path / "extra.zip"
    _zip(extra, {**files, "2_Normalize/": b"", "evil.pkl": b"c"})
    with pytest.raises(IntegrityError):
        safe_extract(extra, spec, tmp_path / "o2")

    trav = tmp_path / "trav.zip"
    _zip(trav, {"../a.json": b"{}"})
    with pytest.raises(IntegrityError):
        safe_extract(trav, {"files": {"../a.json": {"sha256": sha256_bytes(b"{}")}}}, tmp_path / "o3")

    tampered = tmp_path / "tampered.zip"
    _zip(tampered, {"a.json": b"{1}", "sub/b.txt": b"x", "2_Normalize/": b""})
    with pytest.raises(IntegrityError):
        safe_extract(tampered, spec, tmp_path / "o4")


def test_verify_model_dir_detects_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(integrity, "ROOT", tmp_path)
    d = tmp_path / "models/m"
    (d / "e").mkdir(parents=True)
    (d / "w.bin").write_bytes(b"123")
    spec = {"target_dir": "models/m", "files": {"w.bin": {"sha256": sha256_bytes(b"123")}}, "empty_dirs": ["e"]}
    verify_model_dir("m", spec)
    (d / "w.bin").write_bytes(b"124")
    with pytest.raises(IntegrityError):
        verify_model_dir("m", spec)
    (d / "w.bin").write_bytes(b"123")
    (d / "extra.pt").write_bytes(b"")
    with pytest.raises(IntegrityError):
        verify_model_dir("m", spec)


def test_text_hash_ignores_line_endings(tmp_path):
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_bytes(b"x = 1\ny = 2\n")
    b.write_bytes(b"x = 1\r\ny = 2\r\n")
    assert text_sha256(a) == text_sha256(b)


def test_release_manifest_matches_repository():
    integrity.verify_release_manifest()


def test_manifests_are_consistent():
    lock = json.loads((ROOT / "data/manifests/release_data_lock.json").read_text(encoding="utf-8"))
    sm = json.loads((ROOT / "data/manifests/splits_manifest.json").read_text(encoding="utf-8"))
    vp = json.loads((ROOT / "data/manifests/validation_partition.json").read_text(encoding="utf-8"))
    ho = json.loads((ROOT / "data/manifests/qtype_balanced_holdout_v1.json").read_text(encoding="utf-8"))
    from rag_hotpotqa.data.access import partition_validation
    from rag_hotpotqa.data.splits import ids_hash

    for s in ("exploration", "training", "validation"):
        assert ids_hash(sm["split_ids"][s]) == sm["splits"][s]["ids_sha256"] == lock["parts"][s]["ids_sha256"]
    parts = partition_validation(sm["split_ids"]["validation"])
    for p in ("v_select", "v_report"):
        assert ids_hash(parts[p]) == vp["parts"][p]["ids_sha256"] == lock["parts"][p]["ids_sha256"]
    assert ids_hash(ho["ids"]) == ho["ids_sha256"] == lock["parts"]["bc_holdout"]["ids_sha256"]
    assert lock["parts"]["official_dev_distractor"]["ids_sha256"] == sm["splits"]["official_dev"]["ids_sha256"]
    models = json.loads((ROOT / "configs/models.json").read_text(encoding="utf-8"))
    # model files: safe parameter / configuration formats only; license documents by exact top-level name
    allowed = re.compile(r"\.(json|safetensors|txt|npz|npy)$")
    legal = {"LICENSE", "NOTICE", "MODIFICATIONS.md", "DATA_SOURCES.md", "UPSTREAM_MODEL_CARD.md"}
    for name, spec in models["models"].items():
        bad = [f for f in spec["files"] if not (allowed.search(f) or f in legal)]
        assert not bad, (name, bad)
        assert {"LICENSE", "NOTICE", "MODIFICATIONS.md", "DATA_SOURCES.md", "PROVENANCE.json"} <= set(spec["files"]), name
        assert spec["files"][spec["license_file"]] and spec["files"][spec["provenance_file"]], name


PUBLIC = [ROOT / "run.py", *sorted((ROOT / "scripts").rglob("*.py")), *sorted((ROOT / "src").rglob("*.py"))]


def test_public_code_has_no_pickle_loading_or_git_writes():
    # torch.load is allowed only as torch.load(..., weights_only=True): the training scripts resume from checkpoints
    # they wrote themselves (restricted unpickler: tensors / containers / primitives only, no code execution)
    banned = re.compile(r"joblib\.load|pickle\.load|torch\.load\((?![^)]*weights_only=True\))|allow_pickle=True|os\.system|shell=True|"
                        r"\[\s*\"git\"\s*,\s*\"(add|commit|push|tag|remote|config|reset|checkout)\"")
    hits = [f"{p.relative_to(ROOT)}:{i}" for p in PUBLIC for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
            if banned.search(line) and not line.lstrip().startswith("#")]
    assert not hits, hits
