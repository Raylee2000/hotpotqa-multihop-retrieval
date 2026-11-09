"""One-click training: plan / configuration consistency with the research scripts, R5 export, model-manifest override.

No data, models or network are needed (the end-to-end check is `python run.py train --profile smoke`).
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from rag_hotpotqa.release import integrity
from rag_hotpotqa.release.integrity import IntegrityError
from rag_hotpotqa.utils.config import load_yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = load_yaml(ROOT / "configs/training_full.yaml")
HP = CFG["hyperparameters"]


def module(rel: str):
    spec = importlib.util.spec_from_file_location(Path(rel).stem + "_under_test", ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def constants(rel: str) -> dict:
    """Module-level constant assignments (incl. tuple unpacking) of a script, without executing it."""
    out = {}
    for node in ast.parse((ROOT / rel).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            try:
                val = ast.literal_eval(node.value)
            except ValueError:
                continue
            if isinstance(t, ast.Name):
                out[t.id] = val
            elif isinstance(t, ast.Tuple):
                out.update({e.id: v for e, v in zip(t.elts, val)})
    return out


def test_steps_scripts_exist_and_dependencies_run_first():
    tf = module("scripts/train_full.py")
    for profile in CFG["profiles"]:
        cfg, steps = tf.load_plan(profile)
        for s in steps:
            assert (ROOT / s["cmd"][0]).is_file(), s["cmd"][0]
    names = [s["name"] for s in CFG["steps"]]
    assert names == ["general_encoder", "qtype", "qtype_artifact", "sf_fold0", "sf_fold1", "csh_mine", "csh",
                     "candidates_v_select", "candidates_train", "r5", "export", "load_check"]


def test_placeholders_resolve_without_trained_models():
    tf = module("scripts/train_full.py")
    for profile, p in CFG["profiles"].items():
        for s in CFG["steps"]:
            tf.resolve(s["marker"], profile, p["runs"])
            for c in s["cmd"]:
                if "{best." not in c:
                    assert "{" not in tf.resolve(c, profile, p["runs"])


def test_full_profile_uses_research_run_names_and_no_subset_options():
    full = CFG["profiles"]["full"]
    assert full["runs"] == {"general": "hardneg_v1", "qtype": "qtype_v2", "qtype_artifact": "outputs/qtype_artifacts_v1/qtype_v2_frozen_lr",
                            "sf": "sf_cf_v1", "csh": "csh_main_v1", "cand": "general_csh", "r5": "r5_full"}
    assert not full["extra_args"]
    smoke = CFG["profiles"]["smoke"]
    assert all(v.startswith(("smoke_", "outputs/qtype_artifacts_v1/smoke_")) for v in smoke["runs"].values())


def test_general_encoder_config_is_the_formal_one():
    # sha256 of configs/training_v1.yaml recorded in the formal run (outputs/training_v1/hardneg_v1/run_config.json)
    raw = (ROOT / "configs/training_v1.yaml").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw).hexdigest() == "53bedb3894dec2fbae9a6624dd4b7f3a2e097a78731ca4ef75e446a685d7c38c"
    t = load_yaml(ROOT / "configs/training_v1.yaml")
    assert t["variants"][HP["general_encoder"]["variant"]] == {"use_hard": True}
    assert (t["tau"], t["optimizer"]["lr"], t["batch_size"], t["epochs"], t["seed"]) == (0.05, 2e-5, 16, 1, 42)
    assert t["selection"]["validation_subset"] == 1000 and t["selection"]["eval_every_steps"] == 625


def test_recorded_hyperparameters_equal_script_constants():
    sf = constants("scripts/train_sf_selector_cf.py")
    h = HP["sf_selector"]
    assert (sf["LR"], sf["BS"], sf["MAXLEN"], sf["SEED"]) == (h["lr"], h["batch_size"], h["max_length"], h["seed"])
    csh = constants("scripts/train_second_hop.py")
    h = HP["csh"]
    assert (csh["TAU"], csh["LAMBDA"], csh["MARGIN"], csh["BS"], csh["SEED"], csh["MINE_LEN"]) == \
        (h["tau"], h["lambda"], h["margin"], h["batch_size"], h["seed"], h["mining_max_seq_length"])
    assert csh["VARIANTS"][h["variant"]] == {"use_hard": True, "n_hard": 2, "cond": True, "mode": "both"}
    src = (ROOT / "scripts/train_second_hop.py").read_text(encoding="utf-8")
    assert 'ap.add_argument("--k", type=int, default=2' in src and "else (2e-4 if lora else 2e-5)" in src
    gc = constants("scripts/gen_path_candidates.py")
    h = HP["candidates"]
    assert (gc["K_FIRST"], gc["K_SECOND"], gc["K_PAR"]) == (h["k_first"], h["k_second"], h["k_parallel"])
    assert "args.n or 10000" in (ROOT / "scripts/gen_path_candidates.py").read_text(encoding="utf-8")
    # the candidate features of training == those of the public inference (scripts/final_eval/gen_candidates.py)
    assert gc["FEATURES"] == constants("scripts/final_eval/gen_candidates.py")["FEATURES"]
    q = (ROOT / "scripts/train_qtype_classifier.py").read_text(encoding="utf-8")
    assert "for c in (0.1, 1.0, 10.0):" in q and tuple(HP["qtype"]["C_grid"]) == (0.1, 1.0, 10.0)
    assert "LogisticRegression(C=c, max_iter=2000, random_state=42)" in q
    assert "best_c = max(grid, key=lambda c: (grid[c][0][\"macro_f1\"], -c))" in q


def test_r5_grid_and_rule_are_the_original():
    r5 = constants("scripts/train_r5.py")
    assert [list(x) for x in r5["GRID"]] == HP["r5"]["grid"]
    orig = (ROOT / "scripts/train_path_reranker.py").read_text(encoding="utf-8")
    assert '"lm": [(20, 3)] if args.quick else [(100, 3), (200, 4), (300, 3)]' in orig
    assert r5["QUICK_GRID"] == [(20, 3)]
    assert "def fit_lambdamart(d: Data, X, rounds: int, depth: int, lr: float = 0.1):" in orig
    assert "min_samples_leaf=50, random_state=SEED" in orig and "best = max(cands, key=lambda c: (c[1][\"pr1\"], c[1][\"rr\"]))" in orig
    assert "max(cands, key=lambda c: (c[1][\"pr1\"], c[1][\"rr\"]))" in (ROOT / "scripts/train_r5.py").read_text(encoding="utf-8")


class _ToyData:
    """Minimal stand-in for train_path_reranker.Data (groups of candidates, one gold per group)."""

    def __init__(self, rng, n_q=60, n_c=8, n_f=5):
        self.X = rng.normal(size=(n_q * n_c, n_f))
        self.y = np.zeros(n_q * n_c, int)
        self.groups = [np.arange(i * n_c, (i + 1) * n_c) for i in range(n_q)]
        for idx in self.groups:
            self.y[idx[int(np.argmax(self.X[idx, 0] + 0.3 * rng.normal(size=n_c)))]] = 1


def test_r5_export_equals_sklearn_and_reloads(tmp_path):
    from rag_hotpotqa.ranking.r5_trees import R5Trees
    from rag_hotpotqa.training.r5_export import export_r5

    rr = module("scripts/train_path_reranker.py")
    d = _ToyData(np.random.default_rng(1))
    pred = rr.fit_lambdamart(d, d.X, 5, 3)
    free = dict(zip(pred.__code__.co_freevars, (c.cell_contents for c in pred.__closure__)))
    feats = [f"f{i}" for i in range(d.X.shape[1])]
    rep = export_r5(free["trees"], free["lr"], feats, tmp_path / "r5_trees.npz", [("toy", d.X)], n_random=500)
    assert rep["status"] == "PASS"
    m = R5Trees.load(tmp_path / "r5_trees.npz")
    assert m.n_trees == 5 and m.features == feats and np.array_equal(m.predict(d.X), pred(d.X))
    with np.load(tmp_path / "r5_trees.npz", allow_pickle=False) as z:
        assert all(z[k].dtype != object for k in z.files)
    with pytest.raises(FileExistsError):
        export_r5(free["trees"], free["lr"], feats, tmp_path / "r5_trees.npz", [])


def _fake_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(integrity, "ROOT", tmp_path)
    (tmp_path / "configs").mkdir()
    spec = {"target_dir": "models/a", "files": {}, "empty_dirs": []}
    (tmp_path / "configs/models.json").write_text(json.dumps({"models": {"a": spec, "b": {**spec, "target_dir": "models/b"}}}))
    return spec


def test_models_manifest_override(tmp_path, monkeypatch):
    spec = _fake_repo(tmp_path, monkeypatch)
    monkeypatch.delenv(integrity.MODELS_MANIFEST_ENV, raising=False)
    assert integrity.models_manifest_path() == "configs/models.json"
    assert integrity.model_rel("a") == "models/a"
    trained = tmp_path / "outputs/trained_models/p"
    trained.mkdir(parents=True)
    (trained / "models.json").write_text(json.dumps({"models": {"a": {**spec, "target_dir": "outputs/trained_models/p/models/a"},
                                                                "b": {**spec, "target_dir": "outputs/trained_models/p/models/b"}}}))
    for n in ("a", "b"):
        (trained / "models" / n).mkdir(parents=True)
    monkeypatch.setenv(integrity.MODELS_MANIFEST_ENV, "outputs/trained_models/p/models.json")
    assert integrity.model_rel("a") == "outputs/trained_models/p/models/a"
    integrity.verify_models()
    # a manifest with other model names, outside the repository, or pointing outside the repository is refused
    (trained / "models.json").write_text(json.dumps({"models": {"a": {**spec, "target_dir": "outputs/trained_models/p/models/a"}}}))
    with pytest.raises(IntegrityError):
        integrity.verify_models()
    (trained / "models.json").write_text(json.dumps({"models": {"a": {**spec, "target_dir": "../x"}, "b": spec}}))
    with pytest.raises(IntegrityError):
        integrity.model_dir("a")
    monkeypatch.setenv(integrity.MODELS_MANIFEST_ENV, "../elsewhere.json")
    with pytest.raises(IntegrityError):
        integrity.models_manifest_path()


def test_training_scripts_write_no_pickle_model_files():
    for rel in ("scripts/train_sf_selector_cf.py", "scripts/train_r5.py", "scripts/export_qtype_artifact.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        code = src.split("Original docstring:")[-1]
        assert "import joblib" not in code and "joblib.dump" not in code and "pickle.dump" not in code, rel
        assert 'torch.save(head.state_dict(), final / "head.pt")' not in code, rel
    sf = (ROOT / "scripts/train_sf_selector_cf.py").read_text(encoding="utf-8")
    assert 'save_file({k: v.contiguous() for k, v in head.state_dict().items()}, str(final / "head.safetensors"))' in sf


def _fake_training_outputs(root: Path, runs: dict, released: dict) -> None:
    """Training outputs of every model at the paths export_trained_models.py reads (dummy bytes, real file names)."""
    from rag_hotpotqa.utils.artifacts import mark_complete, write_manifest

    lic = set(module("scripts/export_trained_models.py").LICENSE_FILES)
    gen_snap, csh_snap = f"outputs/training_v1/{runs['general']}/snapshot_step00012", f"outputs/csh_v1/{runs['csh']}/snapshot_step00006"
    hist = [{"step": 6, "metrics": {"score": 0.5}}]
    for run_dir, snap in ((f"outputs/training_v1/{runs['general']}", gen_snap), (f"outputs/csh_v1/{runs['csh']}", csh_snap)):
        (root / run_dir).mkdir(parents=True)
        (root / run_dir / "result.json").write_text(json.dumps({"best": {"snapshot": snap, "step": 6, "metrics": {"score": 0.5}},
                                                                "selection_history": hist}))
    r5 = root / "outputs/path_reranker_v1" / runs["r5"]
    r5.mkdir(parents=True)
    (r5 / "result.json").write_text(json.dumps({"hyper": [200, 4], "grid": {}, "v_select": {}}))
    src = {"general_encoder": gen_snap, "csh": csh_snap, "sf_selector_fold0": f"outputs/sf_selector_v2/{runs['sf']}/fold0/final",
           "sf_selector_fold1": f"outputs/sf_selector_v2/{runs['sf']}/fold1/final", "qtype_frozen_lr": runs["qtype_artifact"],
           "r5_full": f"outputs/path_reranker_v1/{runs['r5']}"}
    for name, spec in released.items():
        d = root / src[name]
        for rel in set(spec["files"]) - lic - {"artifact_manifest.json", "COMPLETE.json"}:
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).write_bytes(f"trained {name} {rel}".encode())
        (d / "README.md").write_text("auto-generated, not in the released file set")
    qt = root / runs["qtype_artifact"]
    write_manifest(qt, {p.name: p for p in qt.iterdir() if p.name != "README.md"})
    mark_complete(qt, {"status": "PASS"})


def test_export_trained_models_takes_license_files_from_release_legal(tmp_path, monkeypatch):
    """Real export run (not --dry-run) against the real configs/models.json and release_assets/legal/: training produces
    no license documents, yet the exported file sets equal the released ones and verify through the public loader check."""
    import shutil

    ex = module("scripts/export_trained_models.py")
    released = json.loads((ROOT / "configs/models.json").read_text(encoding="utf-8"))["models"]
    (tmp_path / "configs").mkdir()
    for rel in ("configs/models.json", "configs/training_full.yaml"):
        shutil.copyfile(ROOT / rel, tmp_path / rel)
    shutil.copytree(ROOT / ex.LEGAL_DIR, tmp_path / ex.LEGAL_DIR)
    runs = CFG["profiles"]["smoke"]["runs"]
    _fake_training_outputs(tmp_path, runs, released)
    monkeypatch.setattr(ex, "ROOT", tmp_path)
    monkeypatch.setattr(integrity, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["export_trained_models.py", "--profile", "smoke"])
    assert ex.main() == 0

    man = json.loads((tmp_path / "outputs/trained_models/smoke/models.json").read_text(encoding="utf-8"))
    assert set(man["models"]) == set(released)
    for name, spec in released.items():
        m = man["models"][name]
        assert set(m["files"]) == set(spec["files"]), name          # same file set as the released manifest
        d = tmp_path / m["target_dir"]
        assert not (d / "README.md").exists()
        for rel in set(spec["files"]) & set(ex.LICENSE_FILES):
            b = (d / rel).read_bytes()
            if rel in ex.REWRITTEN:
                assert b != (tmp_path / ex.LEGAL_DIR / name / rel).read_bytes()
            else:                                                     # license / notice texts are the released bytes
                assert integrity.sha256_bytes(b) == spec["files"][rel]["sha256"], (name, rel)
        prov = json.loads((d / "PROVENANCE.json").read_text(encoding="utf-8"))
        assert prov["retrained"] and prov["license"] == "Apache-2.0" and prov["source_run"] == m["source_run"]
        assert set(prov["trained_files"]) == set(spec["files"]) - set(ex.LICENSE_FILES)
        assert all(prov["trained_files"][r]["sha256"] == m["files"][r]["sha256"] for r in prov["trained_files"])
        assert (d / "MODIFICATIONS.md").read_bytes().endswith((tmp_path / ex.LEGAL_DIR / name / "MODIFICATIONS.md").read_bytes())
    monkeypatch.setenv(integrity.MODELS_MANIFEST_ENV, "outputs/trained_models/smoke/models.json")
    integrity.verify_models()                                        # every exported file verifies (file sets + sha256)


def test_export_trained_models_refuses_missing_or_changed_license_files(tmp_path, monkeypatch):
    import shutil

    ex = module("scripts/export_trained_models.py")
    released = json.loads((ROOT / "configs/models.json").read_text(encoding="utf-8"))["models"]
    shutil.copytree(ROOT / ex.LEGAL_DIR, tmp_path / ex.LEGAL_DIR)
    monkeypatch.setattr(ex, "ROOT", tmp_path)
    for name, spec in released.items():
        assert set(ex.released_legal(name, spec)) == set(spec["files"]) & set(ex.LICENSE_FILES)
    (tmp_path / ex.LEGAL_DIR / "csh/NOTICE").write_bytes(b"changed")
    with pytest.raises(SystemExit, match="differs"):
        ex.released_legal("csh", released["csh"])
    (tmp_path / ex.LEGAL_DIR / "csh/NOTICE").unlink()
    with pytest.raises(SystemExit, match="missing"):
        ex.released_legal("csh", released["csh"])
