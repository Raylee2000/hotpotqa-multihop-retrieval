"""Infrastructure: IO safety, resume, crash recovery, concurrency, device fallback, resources, git hygiene."""
import json
import multiprocessing as mp
import shutil
import subprocess
from pathlib import Path

import pytest

from rag_hotpotqa.utils import runtime
from rag_hotpotqa.utils.config import PathSafetyError, safe_path
from rag_hotpotqa.utils.io import ResumableJsonl, atomic_write_json, read_jsonl

PROJECT = Path(__file__).resolve().parents[1]


def test_safe_path_blocks_outside_root(tmp_path):
    assert safe_path(tmp_path / "a" / "b.txt", tmp_path)
    with pytest.raises(PathSafetyError):
        safe_path(tmp_path / ".." / "escape.txt", tmp_path)


def test_windows_unicode_path_and_content(tmp_path):
    target = tmp_path / "結果 資料夾" / "résumé.json"
    atomic_write_json(target, {"問題": "Café — naïve"}, tmp_path)
    assert json.loads(target.read_text(encoding="utf-8")) == {"問題": "Café — naïve"}


def test_resume_dedups_and_skips(tmp_path):
    p = tmp_path / "rows.jsonl"
    s = ResumableJsonl(p, tmp_path)
    assert s.append({"qid": "a", "v": 1}) and not s.append({"qid": "a", "v": 2})
    s2 = ResumableJsonl(p, tmp_path)  # reopen = resume
    assert "a" in s2 and not s2.append({"qid": "a", "v": 3})
    assert [r["v"] for r in read_jsonl(p)] == [1]


def test_partial_trailing_line_is_repaired(tmp_path):
    p = tmp_path / "rows.jsonl"
    p.write_text('{"qid": "a"}\n{"qid": "b", "v"', encoding="utf-8")
    s = ResumableJsonl(p, tmp_path)
    assert set(s.done) == {"a"}
    s.append({"qid": "b", "v": 1})
    assert [r["qid"] for r in read_jsonl(p)] == ["a", "b"]


def test_corrupt_middle_line_raises(tmp_path):
    p = tmp_path / "rows.jsonl"
    p.write_text('{"qid": "a"}\nGARBAGE\n{"qid": "b"}\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        list(read_jsonl(p))


def _writer(args):
    path, root, i = args
    for _ in range(20):
        atomic_write_json(Path(path), {"writer": i, "payload": list(range(200))}, Path(root))
    return i


def test_concurrent_atomic_writes_never_corrupt(tmp_path):
    target = tmp_path / "shared.json"
    with mp.Pool(4) as pool:
        pool.map(_writer, [(str(target), str(tmp_path), i) for i in range(4)])
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["payload"] == list(range(200))
    assert not list(tmp_path.glob("*.tmp"))


def test_cuda_unavailable_falls_back_to_cpu(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch, "xpu"):
        monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    assert runtime.resolve_device("cuda") == "cpu"
    assert runtime.resolve_device("auto") == "cpu"


def test_insufficient_resources_fail_fast(tmp_path):
    with pytest.raises(RuntimeError, match="disk"):
        runtime.check_resources(tmp_path, min_free_gb=10**9, min_free_ram_gb=0)
    with pytest.raises(RuntimeError, match="RAM"):
        runtime.check_resources(tmp_path, min_free_gb=0, min_free_ram_gb=10**9)


def test_seed_reproducibility():
    import numpy as np
    import torch

    runtime.set_seed(42)
    a = (np.random.rand(3), torch.rand(3))
    runtime.set_seed(42)
    b = (np.random.rand(3), torch.rand(3))
    assert (a[0] == b[0]).all() and torch.equal(a[1], b[1])


@pytest.mark.skipif(shutil.which("git") is None or not (PROJECT / ".git").exists(), reason="git repo required")
def test_git_ignores_secrets_and_large_data():
    must_ignore = [".env", "data/raw/x.json", "data/splits/x.txt", "outputs/x.jsonl", ".cache/x",
                   "checkpoints/model.pt", "model.safetensors", "emb.npy"]
    res = subprocess.run(["git", "check-ignore", "--no-index", *must_ignore], cwd=PROJECT,
                         capture_output=True, text=True)
    assert set(res.stdout.split()) == set(must_ignore)
    tracked = subprocess.run(["git", "ls-files"], cwd=PROJECT, capture_output=True, text=True).stdout.split()
    assert ".env" not in tracked
    assert not any(t.startswith(("data/raw/", "data/splits/", "outputs/", ".cache/")) for t in tracked)
    big = [t for t in tracked if (PROJECT / t).exists() and (PROJECT / t).stat().st_size > 5 * 1024 * 1024]
    assert not big, f"large files tracked: {big}"


def test_must_track_manifests_not_ignored():
    if shutil.which("git") is None or not (PROJECT / ".git").exists():
        pytest.skip("git repo required")
    res = subprocess.run(["git", "check-ignore", "--no-index", "data/manifests/splits_manifest.json"],
                         cwd=PROJECT, capture_output=True, text=True)
    assert res.stdout.strip() == ""
