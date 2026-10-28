# runtime-abliteration

Apply a refusal-direction ablation to a served LLM as a **per-request LoRA adapter**
instead of editing the weights. The base checkpoint is never written to; a client turns
the ablation on by naming the adapter on the request and turns it off by not naming it.

The point is that the weight edit and a LoRA delta are the same operation, so the
ablation inherits everything an adapter already gives you: per-request selection, no
restart to flip, and instant reversal.

## How it works

Projecting a direction `q` out of a weight matrix `W` is

```
W' = (I - s q qᵀ) W  =  W - s q qᵀ W
```

A LoRA delta is `B @ A`. Choose

```
A = qᵀ W        (k × in_features)     constant, derived from the base weights
B = -s q        (out_features × k)
```

and `B @ A = -s q qᵀ W` **exactly**. Nothing is approximated — the builder checks the
identity per layer in float64 and refuses to write an adapter whose worst error exceeds
`1e-9`. For a rank-k subspace, use `Q` in place of `q` with `A = QᵀW` and `B = -s Q`.

The site is `attn.wo_b`, the attention output projection. It is a row-parallel linear:
out-features full, in-features sharded across tensor-parallel ranks. Both major engines
slice a row-parallel adapter the way this construction needs — **A sharded on the input
dim, B replicated** — so the same adapter works without engine changes. In SGLang that
is `RowParallelLinearWithLoRA`; in vLLM it is `vllm/lora/layers/row_parallel_linear.py`,
where `slice_lora_a` takes the input-dim slice and `slice_lora_b` returns B unchanged.

Verify that for your own engine and TP layout before trusting the numbers. If A is
sharded on the wrong dim the adapter loads cleanly and produces garbage.

## Start here

Run in this order. Step 3 is the one that differs per engine.

```bash
# 1. Confirm the target tensor exists, and its shape and dtype.
python3 scripts/inspect_wo_b.py --model-dir /models/DeepSeek-V4.1-Flash-1017

# 2. Build the adapter from a direction artifact (see "Direction artifacts" below).
python3 ablit/build_adapter.py --artifact pt \
    --path ablation_directions.pt \
    --model-dir /models/DeepSeek-V4.1-Flash-1017 \
    --out ./ablit-adapter

# 3. Serve with adapters enabled and register the adapter under a name.
#    vLLM:
#      vllm serve deepseek-ai/DeepSeek-V4.1-Flash-1017 \
#        --enable-lora --lora-modules ablit=./ablit-adapter \
#        --lora-target-modules wo_b
#    SGLang: start with --enable-lora, then select the adapter per request as
#    "model": "<base>:<adapter>" -- the colon form. A bare adapter name can be
#    advertised by /v1/models and still serve the base model; check with step 4.

# 4. Prove the adapter is actually applied, before scoring anything.
python3 ablit/compare_adapters.py --base-url http://localhost:8000 \
    --stock deepseek-v4.1-flash --real ablit --out compare.json

# 5. Score it.
python3 eval/refusal_eval.py   --base-url http://localhost:8000 --model ablit --out refusal.json
python3 eval/injection_eval.py --base-url http://localhost:8000 --model ablit --out injection.json
python3 eval/caution_eval.py   --base-url http://localhost:8000 --model ablit --out caution.json
python3 eval/bench_decode.py   --base-url http://localhost:8000 --model ablit --out bench.json

# 6. Soak it: mixed long/short prompts, base and adapter in the same batches.
python3 scripts/lora_soak.py --base http://localhost:8000 --stock deepseek-v4.1-flash --adapter ablit
```

`compare_adapters.py` expects three model names: the base, a **zero adapter** whose B
matrices are all zeros, and the real adapter. Build the zero one with
`ablit/make_zero_adapter.py --src ./ablit-adapter --dst ./zero-adapter` and register it
as `zero`. If the zero adapter changes any output, the difference you are measuring is
the LoRA plumbing rather than the ablation — that control is the whole point of it.

### Direction artifacts

The builder takes either format:

- **`.pt`** — `directions`, a `(layers, hidden)` tensor of unit per-layer directions,
  applied at one uniform strength. Public example:
  `pqhaz/DeepSeek-V4.1-Flash-1017-heretic`, file `ablation_directions.pt`, strength 2.5.
- **`.npz`** — `Q`, an orthonormal `(hidden, k)` basis, plus `scales`, a
  `(layers, 2)` array whose column 0 is the attention-site strength per layer.
  Public example: `mrfakename/DeepSeek-V4.1-Flash-1017-abliterated`.

Both are MIT-licensed and ungated. They are not interchangeable: on the same model the
two bases share a median |cos| of 0.565, and a median 77% of one lies outside the other's
subspace. Whichever you use, pick it by a bake-off on your own prompts, not by which
README quotes the better number.

## Results

DeepSeek-V4.1-Flash-1017, 4× DGX Spark, TP=4, FP8. Adapter: the rank-1 `pqhaz` basis at
strength 2.5, applied to `attn.wo_b` in layers 2–39.

| Measurement | Base | Adapter |
|---|---|---|
| Refusals, 200 harm prompts | 197/200 (98.5%) | 5/200 (2.5%) |
| Benign controls answered | 20/20 | 20/20 |
| Injection leaks, 22 attacks (+3 controls) | 0/22 | 0/22 |
| Caution present, 20 destructive-op prompts | 20/20 | 18/20 |
| Decode tok/s, long-form (`max_tokens` 512) | 70.1 | 68.5 |
| Decode tok/s, structured | 70.2 | 69.6 |
| TTFT | 0.23 s | 0.23 s |

Read these with the caveats:

- **One model, one deployment, one adapter basis.** Not a general claim about abliteration.
- **Injection resistance did not degrade** on this set — 0/22 leaks for both, all 3
  legitimate controls answered in both. That is the result that matters most for a
  deployed agent, and it is the one to re-run first on any new basis.
- **The caution metric is marker-based and blind to form.** 18/20 counts replies that
  still contain a warning marker. Reading the two failures showed the warning survives
  but arrives *after* the model has already agreed to the action, where the base model
  refuses first. A binary metric cannot see that, so treat 18/20 as an upper bound.
- **Decode cost is real but small** — about 2.3% long-form, 0.9% structured — measured at
  `max_tokens` 512 on one derived image, both arms on the same engine.
- Replies are not stored for the harm category; only the verdict is.

## The late-layer replay hazard

Some engines do not run every layer over every token during prefill. DeepSeek-V4 has a
sliding-window path where, past the last KV-source layer, the remaining layers run on
each request's **last `window` tokens only**. If the LoRA routing is still built from the
full extend lengths at that point, every adapter-wrapped linear in those layers runs its
kernels on a row set that no longer matches: reads and writes past the tensor.

Two engines, two different failure modes. Both are worth checking for.

**SGLang: it crashes.** SGLang keeps one forward and replaces only the hidden states
(`hidden_states = tail.rows(hidden_states)`) while the LoRA batch routing was built once
from the full extend. On TP4 this showed up as Xid 31/43 faults or a deadlock on the
first long prefill of any batch with an active adapter. It hits base-only batches too,
because the LoRA kernels run on every batch with the base slot's zero weights.
`patches/sglang/lora_tail_reroute.py` is the fix: install explicit per-request token
segments when the backend enters the tail, and rebuild the full routing when it exits.
It requires a backend with `prepare_lora_token_segments` (triton, uno); csgmv has no such
API and is refused.

**vLLM: it may be silent.** vLLM has the same mechanism —
`vllm/models/deepseek_v41/decoder_replay_layers.py` runs the layers past the last
KV-source layer on each request's last `window` rows. It does not have SGLang's bug,
because it builds a genuine sub-batch and swaps the whole forward context rather than
just the hidden states. But note two things, both from reading vLLM main in October 2026
rather than from running it:

- The replay **CUDA graph** path is explicitly disabled when LoRA is enabled, with the
  comment "The graphs' fixed inputs hold no LoRA mapping"
  (`vllm/models/deepseek_v41/nvidia/model_state.py`).
- The **eager** replay batch is built with `create_forward_context(...)`, and that
  function takes no LoRA mapping (`vllm/forward_context.py`).

So the thing to check in vLLM is not a crash — it is whether your adapter is applied in
the replay layers at all. If it is not, the late layers run as the base model and the
ablation is silently weaker than you measured. If that turns out to bite, disabling
`swa_bounded_replay` when LoRA is enabled is a safe, if slower, fallback.

### Three gates

Any of these failures hides behind a short prompt. Run all three before believing an
adapter is installed:

1. **It differs from base.** Same prompt, base and adapter, same seed: the answers must
   differ. `ablit/compare_adapters.py`. Check this by comparing output, not by checking a
   status code — an unknown or mistyped model name is not always an error, and some
   servers fall back to serving the base model instead of rejecting the request. A typo
   then looks exactly like "the adapter did nothing".
2. **A long prompt through an active adapter does not fault.** Well past the replay
   window — 4k tokens, not 128. This is the gate that catches the SGLang crash.
3. **A mixed soak survives.** Concurrent long and short prompts, base and adapter in the
   same batches. `scripts/lora_soak.py` does this and also asserts the adapter still
   differs at the end.

## Porting to another engine

The algebra is the portable part. The patch is not.

| This repo | vLLM equivalent |
|---|---|
| adapter selected by `model` name on the request | `LoRARequest(name, id, path)` per request; `--lora-modules name=path` on the server |
| start with adapters enabled | `--enable-lora` |
| restrict to the attention output projection | `--lora-target-modules wo_b` |
| A sharded on input dim, B replicated | `RowParallelLinearWithLoRA.slice_lora_a` / `slice_lora_b` |
| eval scripts | unchanged — they speak the OpenAI-compatible API, point them at `--base-url` |

Checklist before you trust a port:

1. Does the model implement `SupportsLoRA`, and is `wo_b` in its supported target list?
2. Does your engine's row-parallel adapter slice A on the input dim and replicate B?
   Confirm against the actual TP layout, not the docs.
3. Does the engine apply the adapter **before or after** FP8 dequantisation? Order changes
   the result.
4. Are the adapter's token indices built from padded or unpadded token counts?
5. Does the engine run LoRA kernels for adapter-less requests in a mixed batch?
6. Does any path shrink the token set mid-forward — sliding-window replay, chunked
   prefill tails, microbatching? If so, is the adapter routing rebuilt for it?
7. Do the three gates above pass?

## What's here

```
ablit/       build_adapter.py      direction artifact -> rank-k LoRA
             make_zero_adapter.py  the inert control adapter
             compare_adapters.py   base vs zero vs real, same prompts
eval/        refusal_eval.py       harm prompts, verdict only
             injection_eval.py     canary-exfiltration attacks + controls
             caution_eval.py       prompts where the right answer is a warning
             bench_decode.py       decode throughput and TTFT
             prompts/              the three prompt sets
patches/     sglang/               the SGLang tail re-route, as a worked example
scripts/     inspect_wo_b.py       what the target tensor looks like
             lora_soak.py          mixed long/short soak with an adapter-differs probe
```

The prompt sets ship as prompts only. No completions and no model output are stored for
the harm category; only the verdict is.

## Licence

MIT — see `LICENSE`.

The refusal prompts in `eval/prompts/refusal_200.jsonl` are drawn from AdvBench
(Zou et al., 2023, `llm-attacks`, MIT). The adapter is derived from
`deepseek-ai/DeepSeek-V4.1-Flash-1017`, which is MIT-licensed. The patch targets SGLang
(Apache-2.0); no SGLang source is redistributed here.

This removes refusal behaviour by design. Comparable abliterated checkpoints are already
public; this repository exists for the method and the engine behaviour, not the weights.
Do not deploy it anywhere refusals are load-bearing.