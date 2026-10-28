"""Re-route SGLang LoRA onto the late-layer tail rows (ABLIT_LORA_TAIL=1).

SGLang-specific. Read this as a worked example of one engine's version of the
failure, not as something to port: an engine that swaps the whole forward context
for its replay sub-batch does not have this problem (see the README, "The late-layer
replay hazard"). This patch exists because SGLang keeps one forward and replaces
only the hidden states, leaving the LoRA routing stale.

DeepSeek-V4 decoder SWA bounded replay runs the layers past the last kv_source
layer over each request's last SWA_WINDOW extend tokens only
(`_forward_layers_hc_pre_from_prev`: `hidden_states = tail.rows(hidden_states)`
after `enter_late_layer_tail`). SGLang builds the LoRA batch routing once per
forward from the FULL extend lengths (`LoRAManager.prepare_lora_batch`), and
nothing updates it at the tail switch. Every LoRA-wrapped linear in a late layer
then runs its kernels with segment lengths summing to the full extend on an
input that holds only the tail rows: reads and writes past the tensor. On TP4
that surfaced as Xid 31/43 faults or a deadlock on the first long prefill of any
batch with an active adapter.

This hits stock-only batches too: triton and csgmv run the LoRA kernels on every
batch (skip_inactive_lora_batches is False), with the base slot's zero weights.

Fix: when the backend enters the tail, install explicit token-row segments (one
per request, its tail length, in request order, which is how the tail metadata
lays the rows out: extend_start_loc = cumsum(tail_lens) - tail_lens); when it
exits, rebuild the full-batch routing. Requires a backend that implements
`prepare_lora_token_segments` (triton, uno). csgmv does not, and is refused.

The LoRA manager belongs to the ModelRunner, which the attention backend cannot
reach, so ModelRunner.forward records the running model's manager for the
duration of the forward (the draft runner has none, so it records None).
"""
import logging
import os

logger = logging.getLogger(__name__)

_current = [None]   # the LoRAManager of the ModelRunner whose forward is running
_rerouted = [False]
_logged = set()


def enabled():
    return os.environ.get("ABLIT_LORA_TAIL", "0").strip() not in ("0", "off", "false", "")


def install_model_runner(module):
    cls = module.ModelRunner
    if getattr(cls.forward, "_dsv41_lora_tail", False):
        return
    original = cls.forward

    def forward(self, forward_batch, *args, **kwargs):
        prev = _current[0]
        _current[0] = getattr(self, "lora_manager", None)
        try:
            return original(self, forward_batch, *args, **kwargs)
        finally:
            _current[0] = prev
            _rerouted[0] = False

    forward._dsv41_lora_tail = True
    cls.forward = forward

    original_init = cls.init_lora_manager

    def init_lora_manager(self, *args, **kwargs):
        out = original_init(self, *args, **kwargs)
        if getattr(self, "lora_manager", None) is not None:
            check_backend(self.lora_manager)
        return out

    cls.init_lora_manager = init_lora_manager
    logger.warning("DSV41 lora_tail: ModelRunner.forward records the LoRA manager")


def _active(forward_batch):
    # Every batch, not only adapter batches: with skip_inactive_lora_batches False
    # (triton, csgmv) the LoRA kernels run on stock-only batches too, so a long stock
    # prompt overruns the tail exactly like an adapter one.
    ids = getattr(forward_batch, "lora_ids", None)
    return list(ids) if ids else None


def install_backend(module):
    cls = module.DeepseekV4AttnBackend
    if getattr(cls.enter_late_layer_tail, "_dsv41_lora_tail", False):
        return
    original_enter = cls.enter_late_layer_tail
    original_exit = cls.exit_late_layer_tail

    def enter_late_layer_tail(self, forward_batch):
        saved = original_enter(self, forward_batch)
        manager = _current[0]
        ids = _active(forward_batch) if manager is not None else None
        if ids is None:
            return saved
        tail = self.tail_forward_metadata.late_layer_tail
        lens = list(tail.local_lens_cpu if tail.cp_metadata is not None
                    else tail.extend_seq_lens_cpu)
        if len(lens) != len(ids):
            raise RuntimeError(
                f"DSV41 lora_tail: {len(lens)} tail segments for {len(ids)} requests")
        pairs = [(i, n) for i, n in zip(ids, lens) if n > 0]
        manager.prepare_lora_token_segments(
            lora_ids=[p[0] for p in pairs], segment_lens=[p[1] for p in pairs])
        _rerouted[0] = True
        key = (len(pairs), sum(lens) < sum(forward_batch.extend_seq_lens_cpu))
        if key not in _logged:
            _logged.add(key)
            logger.warning(
                "DSV41 lora_tail: rerouted LoRA onto %d tail rows (full extend %d, %d requests)",
                sum(lens), sum(forward_batch.extend_seq_lens_cpu), len(pairs))
        return saved

    def exit_late_layer_tail(self, saved, forward_batch):
        original_exit(self, saved, forward_batch)
        if _rerouted[0]:
            _rerouted[0] = False
            _current[0].prepare_lora_batch(forward_batch)

    enter_late_layer_tail._dsv41_lora_tail = True
    cls.enter_late_layer_tail = enter_late_layer_tail
    cls.exit_late_layer_tail = exit_late_layer_tail
    logger.warning("DSV41 lora_tail: late-layer tail re-routes LoRA segments")


def check_backend(lora_manager):
    """Refuse a backend that cannot take explicit token segments, at init."""
    backend = lora_manager.lora_backend
    impl = type(backend).prepare_lora_token_segments
    from sglang.srt.lora.backend.base_backend import BaseLoRABackend
    if impl is BaseLoRABackend.prepare_lora_token_segments:
        raise RuntimeError(
            f"ABLIT_LORA_TAIL needs a LoRA backend with explicit token segments; "
            f"{type(backend).__name__} has none. Use --lora-backend triton.")
