#!/usr/bin/env python3
"""Build a ZERO control adapter from a real one.

Purpose: prove the LoRA plumbing is inert. If an adapter whose B matrices are all
zeros changes any output, the difference is the plumbing, not the ablation. Run it
BEFORE trusting a refusal-rate change from the real adapter.

`delta = B @ A`, so zeroing B makes the delta exactly zero while keeping the same
shapes, ranks, layer coverage and loader path as the real adapter.

Usage:
    python3 ablit/make_zero_adapter.py --src ./ablit-adapter --dst ./zero-adapter

Needs torch and safetensors. They are imported inside main() so that `--help`
works on a machine that has none of them.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="adapter directory to copy from")
    ap.add_argument("--dst", required=True, help="output directory for the zero adapter")
    a = ap.parse_args()

    # Heavy imports AFTER parse_args: --help exits above.
    import torch
    from safetensors.torch import load_file, save_file

    os.makedirs(a.dst, exist_ok=True)
    shutil.copy(os.path.join(a.src, "adapter_config.json"),
                os.path.join(a.dst, "adapter_config.json"))

    sd = load_file(os.path.join(a.src, "adapter_model.safetensors"))
    out, zeroed = {}, 0
    for k, v in sd.items():
        if "lora_B" in k:
            out[k] = torch.zeros_like(v)
            zeroed += 1
        else:
            out[k] = v
    save_file(out, os.path.join(a.dst, "adapter_model.safetensors"))

    # A zero delta must be provable from the file, not assumed.
    worst = 0.0
    for k in sd:
        if "lora_B" in k:
            worst = max(worst, float(out[k].abs().max()))
    meta = {
        "kind": "zero-control-adapter",
        "source": a.src,
        "tensors": len(sd),
        "lora_B_zeroed": zeroed,
        "max_abs_B_after": worst,
    }
    json.dump(meta, open(os.path.join(a.dst, "ZERO_REPORT.json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))
    if worst != 0.0:
        raise SystemExit("zeroing failed: a lora_B tensor is non-zero")


if __name__ == "__main__":
    main()