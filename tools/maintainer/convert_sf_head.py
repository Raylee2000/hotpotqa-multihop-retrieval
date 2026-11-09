"""MAINTAINER-ONLY, one-time conversion of an SF-selector head (head.pt, a plain torch state_dict) to safetensors.

Not part of the public workflow. torch.load is used with weights_only=True (restricted unpickler: tensors and
containers only). The converted tensors must be bit-identical to the originals or nothing is written.
Usage: python tools/maintainer/convert_sf_head.py <head.pt> <head.safetensors>
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main() -> int:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    if dst.exists():
        raise SystemExit(f"{dst} exists")
    sd = torch.load(src, map_location="cpu", weights_only=True)
    if set(sd) != {"weight", "bias"} or tuple(sd["weight"].shape) != (1, 384) or tuple(sd["bias"].shape) != (1,):
        raise SystemExit(f"unexpected head layout: { {k: tuple(v.shape) for k, v in sd.items()} }")
    save_file({k: v.contiguous() for k, v in sd.items()}, str(dst))
    back = load_file(str(dst))
    if set(back) != set(sd) or not all(torch.equal(back[k], sd[k]) and back[k].dtype == sd[k].dtype for k in sd):
        dst.unlink()
        raise SystemExit("converted tensors differ")
    print("PASS", src.name, "->", dst.name, {k: (tuple(v.shape), str(v.dtype)) for k, v in sd.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
