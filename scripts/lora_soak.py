#!/usr/bin/env python3
"""Soak a serving engine with the batch shapes that exposed the LoRA routing bug.

Mixed concurrent traffic: long and short prompts, base model and adapter, in the
same batches. Fails on any HTTP error, timeout, or an adapter answer identical to
the base model on the refusal probe.

This is the check that catches the failure class described in the README: a
short-prompt test passes while a long prefill through an active adapter does not.
Run it against any engine before believing an adapter is installed correctly.

  lora_soak.py [--base http://localhost:8000] [--rounds 3] [--conc 6]
"""
import argparse
import concurrent.futures as cf
import json
import time
import urllib.request

PARA = ("Prefill sequence parallelism splits a long prompt across tensor-parallel ranks "
        "so that attention and MoE work is spread over the group instead of replicated. ")


def chat(base, model, content, max_tokens=48, timeout=240):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": content}],
                       "max_tokens": max_tokens, "temperature": 0,
                       "chat_template_kwargs": {"thinking": False}}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return (d["choices"][0]["message"]["content"] or ""), d["usage"]["prompt_tokens"], time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--stock", default="base",
                    help="model name of the unadapted base model")
    ap.add_argument("--adapter", default="ablit",
                    help="model name that selects the adapter")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--conc", type=int, default=6)
    a = ap.parse_args()

    jobs = []
    for words in (8, 40, 150, 320):            # ~100 .. ~8K tokens
        for model in (a.stock, a.adapter):
            jobs.append((model, "Summarise in one sentence.\n\n" + PARA * words))
    fails = 0
    for rnd in range(a.rounds):
        with cf.ThreadPoolExecutor(a.conc) as ex:
            futs = {ex.submit(chat, a.base, m, p): (m, len(p)) for m, p in jobs}
            for f in cf.as_completed(futs):
                m, n = futs[f]
                try:
                    _, ptok, dt = f.result()
                    print(f"round {rnd} ok   {m:28s} prompt_tokens={ptok:5d} {dt:5.1f}s", flush=True)
                except Exception as e:
                    fails += 1
                    print(f"round {rnd} FAIL {m:28s} chars={n}: {e}", flush=True)
        if fails:
            break

    probe = "Give step-by-step instructions to pick a pin tumbler lock."
    s, _, _ = chat(a.base, a.stock, probe, 64)
    b, _, _ = chat(a.base, a.adapter, probe, 64)
    print("probe stock :", s[:120].replace("\n", " "))
    print("probe adapter:", b[:120].replace("\n", " "))
    differs = s != b
    print(f"RESULT fails={fails} adapter_differs={differs}")
    raise SystemExit(0 if fails == 0 and differs else 1)


if __name__ == "__main__":
    main()