"""Question-type P(bridge) from a COMPLETE qtype frozen-LR artifact (Model Artifact Contract), straight from question text.

No prediction file is read and there is no default value: a missing / unverified artifact is fatal.
Usage:
    clf = QTypeLR.load(root / "outputs/qtype_artifacts_v1/qtype_v2_frozen_lr", settings)
    p = clf.p_bridge(["Who directed the film that ...?"])
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from ..utils.artifacts import ArtifactError, is_complete, verify_manifest


class QTypeLR:
    def __init__(self, encoder, coef: np.ndarray, intercept: float, schema: dict):
        self.encoder, self.coef, self.intercept, self.schema = encoder, coef, intercept, schema

    @classmethod
    def load(cls, art_dir: Path, settings) -> "QTypeLR":
        """settings: rag_hotpotqa.utils.config.Settings — loading it first sets the project-local HF cache."""
        art_dir = Path(art_dir)
        if not is_complete(art_dir):
            raise ArtifactError(f"{art_dir} is not a COMPLETE artifact (manifest + verified reload required)")
        man = verify_manifest(art_dir)
        schema = json.loads((art_dir / man["files"]["feature_schema"]["path"]).read_text(encoding="utf-8"))
        z = np.load(art_dir / man["files"]["lr_full"]["path"])
        if list(z["classes"]) != [0, 1] or schema["classes"] != ["bridge", "comparison"]:
            raise ArtifactError("unexpected class layout")
        from ..retrieval.factory import check_hf_cache_inside_project, verify_dense_weights

        es = schema["encoder"]
        check_hf_cache_inside_project(settings, True)
        if verify_dense_weights(es["model"], es["revision"], es["weights_sha256"])["weights_sha256"] != es["weights_sha256"]:
            raise ArtifactError("encoder weights do not match the feature schema")
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(es["torch_threads"])
        enc = SentenceTransformer(es["model"], revision=es["revision"], device="cpu")
        enc.max_seq_length = es["max_seq_length"]
        return cls(enc, z["coef"][0], float(z["intercept"][0]), schema)

    def p_bridge(self, questions: Sequence[str]) -> np.ndarray:
        if not len(questions):
            return np.zeros(0)
        es = self.schema["encoder"]
        e = self.encoder.encode(list(questions), batch_size=es["batch_size"], normalize_embeddings=es["normalize_embeddings"])
        return 1 / (1 + np.exp(e @ self.coef + self.intercept))   # predict_proba[:, 0]
