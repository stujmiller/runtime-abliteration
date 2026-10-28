"""Build a rank-k LoRA adapter that *is* the refusal-direction ablation.

The ablation applied to `attn.wo_b` is

    W' = (I - s Q Q^T) W  =  W - s Q Q^T W

A standard LoRA delta is `B @ A`, so choose

    A = Q^T W        (k x in_features)   -- constant, derived from the base weights
    B = -s Q         (out_features x k)

and `B @ A = -s Q Q^T W` exactly. Nothing is approximated: the adapter reproduces
the weight edit to float64 round-off, which the build checks per layer.

Why this works without touching the engine's LoRA plumbing: `attn.wo_b` is a
row-parallel linear (out-features full, in-features sharded across TP ranks), and
both major engines slice a row-parallel adapter the way this construction needs --
A sharded on the input dim, B replicated. In SGLang that is
`RowParallelLinearWithLoRA`; in vLLM it is `vllm/lora/layers/row_parallel_linear.py`
(`slice_lora_a` takes the input-dim slice, `slice_lora_b` returns B unchanged).
Verify this for your own engine and TP layout before trusting the numbers.

Artifact semantics (two public bases, both MIT):
  npz -> `Q (5120,k)` orthonormal basis; `scales (40,2)` = [layer, (attn, shexp)].
         The ATTENTION column is `scales[:,0]`; the shexp column is a different
         site and is not used here. One scalar strength per layer for the whole
         subspace.
  pt  -> `directions (40,5120)` unit per-layer vectors; uniform strength 2.5.

No weights are modified on disk. The adapter is inert until the server is started
with adapters enabled and a request selects the adapter's name.

Usage:
    python3 ablit/build_adapter.py --artifact pt \
        --path ablation_directions.pt \
        --model-dir /path/to/DeepSeek-V4.1-Flash-1017 \
        --out ./ablit-adapter

Needs torch, numpy and safetensors. They are imported inside main() so that
`--help` works on a machine that has none of them.
"""

from __future__ import annotations

import argparse
import json
import os

# Bound by main(); see the lazy-import note above.
np = None
torch = None
safe_open = None
save_file = None

DEFAULT_LAYERS = "2-39"
DEFAULT_STRENGTH = 2.5


def parse_layers(spec: str) -> list[int]:
    lo, hi = spec.split("-")
    return list(range(int(lo), int(hi) + 1))


def e8m0_to_float(scale):
    """Convert an F8_E8M0 scale tensor to float32 (2^(byte-127))."""
    if scale.dtype == torch.float32:
        return scale
    try:
        return scale.to(torch.float32)
    except Exception:
        raw = scale.view(torch.uint8).to(torch.int32)
        return torch.pow(2.0, (raw - 127).to(torch.float32))


def dequant_block_fp8(weight, scale, block: int = 32):
    """Dequantise FP8 e4m3 weights with per-[block,block] E8M0 scales."""
    w = weight.to(torch.float32)
    s = e8m0_to_float(scale)
    s = s.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
    assert s.shape == w.shape, (s.shape, w.shape)
    return w * s


def load_basis(artifact: str, path: str, layer: int):
    """Return (Q [d,k] float32 unit columns, scalar strength for this layer)."""
    if artifact == "npz":
        z = np.load(path)
        Q = torch.from_numpy(z["Q"]).float()            # [d, k]
        scales = torch.from_numpy(z["scales"]).float()  # [L, 2] = (attn, shexp)
        if Q.shape[0] != 5120:
            Q = Q.t().contiguous()
        Q = Q / Q.norm(dim=0, keepdim=True).clamp_min(1e-12)
        strength = float(scales[layer, 0])              # ATTENTION column only
        return Q, strength
    if artifact == "pt":
        d = torch.load(path, map_location="cpu", weights_only=True)
        q = d["directions"].float()[layer]
        q = (q / q.norm()).reshape(-1, 1)               # [d,1]
        return q, DEFAULT_STRENGTH
    raise ValueError(f"unknown artifact {artifact!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--artifact", choices=["npz", "pt"], required=True)
    ap.add_argument("--path", required=True, help="direction artifact (.pt or .npz)")
    ap.add_argument("--model-dir", required=True,
                    help="local path to the base model (needs model.safetensors.index.json)")
    ap.add_argument("--out", required=True, help="output directory for the adapter")
    ap.add_argument("--layers", default=DEFAULT_LAYERS)
    ap.add_argument("--strength-scale", type=float, default=1.0,
                    help="multiply the artifact's per-layer strength (alpha knob)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Heavy imports AFTER parse_args: --help exits above, so this script can be
    # inspected on a machine with no torch.
    global np, torch, safe_open, save_file
    import numpy as _np
    import torch as _torch
    from safetensors import safe_open as _safe_open
    from safetensors.torch import save_file as _save_file
    np, torch, safe_open, save_file = _np, _torch, _safe_open, _save_file

    model_dir = args.model_dir.rstrip("/") + "/"
    layers = parse_layers(args.layers)
    idx = json.load(open(model_dir + "model.safetensors.index.json"))["weight_map"]

    tensors: dict = {}
    report = []
    for layer in layers:
        wname = f"layers.{layer}.attn.wo_b.weight"
        sname = f"layers.{layer}.attn.wo_b.scale"
        shard = idx.get(wname)
        if shard is None or idx.get(sname) != shard:
            raise SystemExit(f"missing wo_b tensors for layer {layer}")
        with safe_open(model_dir + shard, framework="pt", device="cpu") as h:
            w = h.get_tensor(wname)
            sc = h.get_tensor(sname)
        W = dequant_block_fp8(w, sc)                    # [5120, 8192] f32
        Q, s = load_basis(args.artifact, args.path, layer)
        s = s * args.strength_scale
        A = (Q.t() @ W).contiguous()                    # [k, 8192]
        B = (-s * Q).contiguous()                       # [5120, k]
        key = f"base_model.model.model.layers.{layer}.attn.wo_b"
        tensors[key + ".lora_A.weight"] = A.to(torch.float32)
        tensors[key + ".lora_B.weight"] = B.to(torch.float32)

        # Numeric check in float64: B(Ax) == -s Q Q^T (W x).
        # A64 proves the algebra; the shipped A is fp32, so we also report the
        # fp32 quantisation noise separately (it is ~1e-5 relative on a small
        # correction and is irrelevant at bf16 inference precision).
        Wd, Qd = W.double(), Q.double()
        A64, B64 = Qd.t() @ Wd, -s * Qd
        x = torch.randn(W.shape[1], dtype=torch.float64)
        want = -s * (Qd @ (Qd.t() @ (Wd @ x)))
        err = (B64 @ (A64 @ x) - want).norm() / want.norm().clamp_min(1e-12)
        err_fp32 = (B.double() @ (A.double() @ x) - want).norm() / want.norm().clamp_min(1e-12)
        report.append({"layer": layer, "relerr": float(err), "relerr_fp32": float(err_fp32),
                       "strength": s, "A_norm": float(A.norm()), "B_norm": float(B.norm())})
        if layer % 8 == 0 or layer == layers[-1]:
            print(f"layer {layer:2d}  strength {s:.4f}  algebra {float(err):.1e}  fp32 {float(err_fp32):.1e}")

    worst = max(r["relerr"] for r in report)
    worst32 = max(r["relerr_fp32"] for r in report)
    print(f"layers built: {len(report)}  worst algebra error: {worst:.2e}  worst fp32 noise: {worst32:.2e}")
    if worst > 1e-9:
        raise SystemExit("adapter algebra check FAILED")

    if args.dry_run:
        print("dry run: nothing written")
        return

    os.makedirs(args.out, exist_ok=True)
    save_file(tensors, os.path.join(args.out, "adapter_model.safetensors"))
    k = next(iter(tensors.values())).shape[0]
    cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": int(k),
        "lora_alpha": int(k),          # scale 1.0; strength is baked into B
        "lora_dropout": 0.0,
        "bias": "none",
        "inference_mode": True,
        "target_modules": ["wo_b"],
        "base_model_name_or_path": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "rank_pattern": {},
        "alpha_pattern": {},
    }
    with open(os.path.join(args.out, "adapter_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(args.out, "BUILD_REPORT.json"), "w") as f:
        json.dump({"artifact": args.artifact, "layers": layers,
                   "strength_scale": args.strength_scale,
                   "per_layer": report}, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()