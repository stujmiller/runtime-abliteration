#!/usr/bin/env python3
"""compare_adapters.py — is the LoRA plumbing inert?

Runs the same prompts through several served model names at temperature 0 and
compares outputs exactly:

  stock vs ZERO   must be IDENTICAL. A zero adapter's delta is B@A = 0, so any
                  difference here is plumbing (sharding, offsets, buffer reuse),
                  not ablation.
  stock vs REAL   is expected to differ. How much is the ablation's effect.

Without the zero control, a changed refusal rate cannot be attributed to the
direction rather than to the machinery that applies it.

    compare_adapters.py --base-url http://localhost:8000 \
        --stock deepseek-v4.1-flash --zero zero --real ablit \
        --prompts eval/prompts/refusal_set.jsonl --out eval/results/zero-control.json
"""

from __future__ import annotations

import argparse
import json
import urllib.request


def ask(base, model, prompt, max_tokens, timeout=300):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"thinking": False},
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        obj = json.loads(r.read().decode())
    ch = (obj.get("choices") or [{}])[0]
    return (ch.get("message") or {}).get("content") or "", obj.get("usage") or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--stock", default="deepseek-v4.1-flash")
    ap.add_argument("--zero", default="zero")
    ap.add_argument("--real", default="ablit")
    ap.add_argument("--prompts", default="eval/prompts/refusal_set.jsonl")
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    items = [json.loads(l) for l in open(a.prompts) if l.strip()]
    rows = []
    zero_identical = zero_differ = 0
    real_differ = 0

    for it in items:
        outs = {}
        for name, model in (("stock", a.stock), ("zero", a.zero), ("real", a.real)):
            text, usage = ask(a.base_url, model, it["prompt"], a.max_tokens)
            outs[name] = {"text": text, "tokens": usage.get("completion_tokens")}
        same_zero = outs["stock"]["text"] == outs["zero"]["text"]
        same_real = outs["stock"]["text"] == outs["real"]["text"]
        zero_identical += same_zero
        zero_differ += (not same_zero)
        real_differ += (not same_real)
        rows.append({
            "id": it["id"], "category": it["category"],
            "stock_eq_zero": same_zero, "stock_eq_real": same_real,
            "stock_tokens": outs["stock"]["tokens"],
            "zero_tokens": outs["zero"]["tokens"],
            "real_tokens": outs["real"]["tokens"],
            "stock_head": " ".join(outs["stock"]["text"].split())[:90],
            "real_head": " ".join(outs["real"]["text"].split())[:90],
        })
        flag = "IDENTICAL" if same_zero else "*** DIFFERS ***"
        print(f"{it['id']:>4} stock-vs-zero: {flag:<17} stock-vs-real: "
              f"{'same' if same_real else 'differs'}")

    n = len(items)
    doc = {
        "kind": "adapter-zero-control",
        "models": {"stock": a.stock, "zero": a.zero, "real": a.real},
        "max_tokens": a.max_tokens,
        "summary": {
            "n": n,
            "stock_eq_zero": zero_identical,
            "stock_ne_zero": zero_differ,
            "stock_eq_real": n - real_differ,
            "stock_ne_real": real_differ,
        },
        "rows": rows,
    }
    json.dump(doc, open(a.out, "w"), indent=2)
    print(f"\nstock == zero : {zero_identical}/{n}  (must be {n}/{n})")
    print(f"stock != real : {real_differ}/{n}")
    print(f"wrote {a.out}")
    if zero_differ:
        print("\nWARNING: the zero adapter changed outputs — investigate the LoRA "
              "plumbing before attributing anything to the ablation.")


if __name__ == "__main__":
    main()