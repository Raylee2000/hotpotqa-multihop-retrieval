"""Minimal LoRA for the pinned MiniLM (BertModel inside SentenceTransformer) — no external dependency.

    y = W x + b + (alpha / r) · B A dropout(x),   A ∈ R^{r×in} (Kaiming-uniform), B ∈ R^{out×r} (zeros)

Base weights are frozen (requires_grad False); only A, B train. Because B starts at zero the adapted model is
exactly the base model at step 0. `merged_copy` folds B A into W and returns a plain SentenceTransformer
(same modules, same mean pooling + Normalize), so evaluation/inference code is unchanged.
Target presets name the BertModel Linear modules by suffix:
    qv   attention.self.query, attention.self.value                     (the original LoRA paper's choice)
    qkvo + attention.self.key, attention.output.dense
    all  + intermediate.dense, output.dense (FFN; the attention output is matched by 'attention.output.dense')
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn

TARGETS = {
    "qv": ("attention.self.query", "attention.self.value"),
    "qkvo": ("attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense"),
    "all": ("attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense",
            "intermediate.dense", "layer.{i}.output.dense"),
}


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.r, self.scale = r, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.lora_A.T @ self.lora_B.T) * self.scale

    def merged(self) -> nn.Linear:
        # skip_init: no random initialisation, so merging never consumes the global torch RNG (resume determinism)
        out = nn.utils.skip_init(nn.Linear, self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        with torch.no_grad():
            out.weight.copy_(self.base.weight + self.scale * self.lora_B @ self.lora_A)
            if self.base.bias is not None:
                out.bias.copy_(self.base.bias)
        return out


def _matches(name: str, preset: str) -> bool:
    for t in TARGETS[preset]:
        if "{i}" in t:  # FFN output: 'encoder.layer.<i>.output.dense', not 'attention.output.dense'
            parts = name.split(".")
            if len(parts) >= 4 and parts[-2:] == ["output", "dense"] and parts[-3].isdigit():
                return True
        elif name.endswith(t):
            return True
    return False


def apply_lora(st_model, preset: str, r: int, alpha: float, dropout: float) -> list[str]:
    """Freeze every parameter of the SentenceTransformer and wrap the target Linear layers in place."""
    for p in st_model.parameters():
        p.requires_grad_(False)
    bert = st_model[0].auto_model
    names = [n for n, m in bert.named_modules() if isinstance(m, nn.Linear) and _matches(n, preset)]
    for n in names:
        parent_name, _, child = n.rpartition(".")
        parent = bert.get_submodule(parent_name)
        setattr(parent, child, LoRALinear(getattr(parent, child), r, alpha, dropout))
    return names


def adapter_state(st_model) -> dict:
    return {k: v.detach().clone() for k, v in st_model.state_dict().items() if ".lora_" in k}


def load_adapter_state(st_model, state: dict) -> None:
    own = {k for k in st_model.state_dict() if ".lora_" in k}
    if own != set(state):
        raise ValueError(f"adapter keys differ: missing {sorted(own - set(state))[:3]} unexpected {sorted(set(state) - own)[:3]}")
    st_model.load_state_dict(state, strict=False)


def merged_copy(st_model):
    """A plain SentenceTransformer with every LoRALinear folded into its base Linear."""
    m = copy.deepcopy(st_model)
    bert = m[0].auto_model
    for n, mod in list(bert.named_modules()):
        if isinstance(mod, LoRALinear):
            parent_name, _, child = n.rpartition(".")
            setattr(bert.get_submodule(parent_name), child, mod.merged())
    for p in m.parameters():
        p.requires_grad_(True)
    return m
