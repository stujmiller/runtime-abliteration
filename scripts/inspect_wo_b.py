"""Inspect the wo_b tensors of a DeepSeek-V4.1-Flash checkpoint.

Reports, for one sample layer and then all layers: shard file, dtype, shape, and
the block-quant scale shape. This is the input the LoRA adapter builder needs --
`A = Q^T W` requires dequantising W (FP8 e4m3 with E8M0 [32,32] block scales)
before multiplying by the basis.

Run this first if you are porting to a new engine or a different checkpoint: it
tells you whether `attn.wo_b` exists under that name, what shape it has, and
therefore whether A must be sliced on the input dim for your TP layout.

Usage:
    python3 scripts/inspect_wo_b.py --model-dir /path/to/DeepSeek-V4.1-Flash-1017

Needs safetensors. It is imported inside main() so that `--help` works on a
machine that does not have it.
"""

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-dir", required=True,
                    help="local path to the base model (needs model.safetensors.index.json)")
    ap.add_argument("--layer", type=int, default=20, help="sample layer to inspect")
    args = ap.parse_args()

    from safetensors import safe_open

    base = args.model_dir.rstrip("/") + "/"
    idx = json.load(open(base + "model.safetensors.index.json"))["weight_map"]

    sample = f"layers.{args.layer}.attn.wo_b.weight"
    print("sample:", sample)
    for name in (sample, sample.replace(".weight", ".scale"),
                 f"layers.{args.layer}.attn.wo_a.weight"):
        shard = idx.get(name)
        print(" ", name, "->", shard)
        if not shard:
            continue
        with safe_open(base + shard, framework="pt", device="cpu") as h:
            sl = h.get_slice(name)
            print("     shape", sl.get_shape(), "dtype", sl.get_dtype())

    # Which layers carry wo_b, and do any carry a scale of a different shape?
    rows = []
    for k in idx:
        if k.endswith("attn.wo_b.weight"):
            rows.append(int(k.split(".")[1]))
    print("layers with wo_b.weight:", sorted(rows))

    # Distinct shapes/dtypes across all layers (should be uniform)
    shapes = {}
    for layer in sorted(rows):
        for suffix in ("weight", "scale"):
            name = f"layers.{layer}.attn.wo_b.{suffix}"
            shard = idx.get(name)
            if not shard:
                shapes[(suffix, "MISSING")] = shapes.get((suffix, "MISSING"), 0) + 1
                continue
            with safe_open(base + shard, framework="pt", device="cpu") as h:
                sl = h.get_slice(name)
                key = (suffix, tuple(sl.get_shape()), sl.get_dtype())
            shapes[key] = shapes.get(key, 0) + 1
    print("distinct (suffix, shape, dtype) -> count:")
    for k, v in sorted(shapes.items(), key=lambda x: str(x[0])):
        print("  ", k, "->", v)


if __name__ == "__main__":
    main()