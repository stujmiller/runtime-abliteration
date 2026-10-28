#!/usr/bin/env python3
"""bench_decode.py — streaming decode benchmark + speculative-accept sampling.

Measures what boot #2 must not regress:
  * decode tok/s   — inter-token rate after the first token, so prefill does not
                     pollute the number;
  * TTFT           — time to first token;
  * DSpark accept  — sglang:spec_accept_rate / spec_accept_length from /metrics.

Stdlib only: runs from any host that can reach the endpoint.

    bench_decode.py --base-url http://localhost:8000 --out baseline.json [--repeats 3]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request

PROMPTS = {
    "long_form": (
        "Write a detailed technical explanation of how TCP congestion control works. "
        "Cover slow start, congestion avoidance, fast retransmit and fast recovery, "
        "and explain how the algorithms interact. Use headings."
    ),
    "structured": (
        "Explain the differences between a process and a thread, then describe five "
        "concrete situations where each is the better choice, with a short rationale "
        "for every situation."
    ),
}


def post_stream(url, payload, timeout=300):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    deltas, t_first, t_last, usage = 0, None, None, None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    deltas += 1
    t_end = time.perf_counter()
    if t_first is None or t_last is None:
        raise RuntimeError("no content deltas received")
    decode_s = t_last - t_first
    # The SSE stream COALESCES several tokens per delta (observed ~2.7:1), so a
    # delta count is not a token count. Prefer the authoritative usage figure.
    ctoks = (usage or {}).get("completion_tokens") or deltas
    return {
        "ttft_s": round(t_first - t0, 4),
        "decode_s": round(decode_s, 4),
        "deltas": deltas,
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "tokens_per_delta": round(ctoks / deltas, 3) if deltas else None,
        "decode_tok_s": round((ctoks - 1) / decode_s, 3) if decode_s > 0 and ctoks > 1 else None,
        "delta_per_s": round((deltas - 1) / decode_s, 3) if decode_s > 0 and deltas > 1 else None,
        "e2e_tok_s": round(ctoks / (t_end - t0), 3),
    }


def metrics(base):
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=10) as r:
            text = r.read().decode()
    except Exception as exc:  # endpoint may be mid-restart
        return {"error": str(exc)}
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        for name in ("sglang:spec_accept_rate", "sglang:spec_accept_length",
                     "sglang:spec_verify_calls_total", "sglang:num_requests_total"):
            if line.startswith(name):
                try:
                    out[name] = float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    url = a.base_url.rstrip("/") + "/v1/chat/completions"
    before = metrics(a.base_url)
    runs = {}
    for name, prompt in PROMPTS.items():
        results = []
        for i in range(a.repeats):
            payload = {
                "model": a.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": a.max_tokens,
                "temperature": 0,
                "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"thinking": False},
            }
            r = post_stream(url, payload)
            # The accept gauges are workload-dependent, so sample them right
            # after each run and aggregate — an idle reading is not comparable.
            r["metrics"] = metrics(a.base_url)
            results.append(r)
            print(f"{name} run {i+1}: {r['decode_tok_s']} tok/s  ttft {r['ttft_s']}s  "
                  f"tokens {r['completion_tokens']}  deltas {r['deltas']}")
        runs[name] = {
            "runs": results,
            "decode_tok_s_median": statistics.median(
                [x["decode_tok_s"] for x in results if x["decode_tok_s"]]),
            "ttft_s_median": statistics.median([x["ttft_s"] for x in results]),
        }
    after = metrics(a.base_url)

    doc = {
        "kind": "decode-benchmark",
        "model": a.model,
        "base_url": a.base_url,
        "max_tokens": a.max_tokens,
        "repeats": a.repeats,
        "runs": runs,
        "metrics_before": before,
        "metrics_after": after,
    }
    json.dump(doc, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    for name, r in runs.items():
        print(f"  {name}: median {r['decode_tok_s_median']} tok/s, ttft {r['ttft_s_median']}s")
    print(f"  accept_rate={after.get('sglang:spec_accept_rate')} "
          f"accept_length={after.get('sglang:spec_accept_length')}")


if __name__ == "__main__":
    main()