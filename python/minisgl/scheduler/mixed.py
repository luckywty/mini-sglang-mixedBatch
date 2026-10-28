from __future__ import annotations

from typing import TYPE_CHECKING

from minisgl.core import Batch, BatchLayout

if TYPE_CHECKING:
    from .decode import DecodeManager
    from .prefill import PrefillManager


def schedule_mixed_batch(
    decode_manager: DecodeManager,
    prefill_manager: PrefillManager,
    total_token_budget: int,
) -> Batch | None:
    """Select one packed batch; allocation/forward are separate scheduler steps.

    All active decodes are retained, even if they exceed the budget. In that case
    the iteration is decode-only and its effective budget is the decode count.
    Prefill admission keeps the existing slots, prefix locks and KV reservations.
    """
    if total_token_budget <= 0:
        raise ValueError("Mixed batching requires a positive total token budget")

    decode = decode_manager.schedule_next_batch()
    decode_reqs = decode.reqs if decode is not None else []
    # Validate before prefill admission mutates slots, locks or the pending queue.
    if any(req.extend_len != 1 or not req.needs_sampling for req in decode_reqs):
        raise ValueError("Active decode requests must each compute one token and sample")

    remaining_budget = max(0, total_token_budget - len(decode_reqs))
    prefill = prefill_manager.schedule_next_batch(remaining_budget) if remaining_budget else None
    prefill_reqs = prefill.reqs if prefill is not None else []
    if not decode_reqs and not prefill_reqs:
        return None

    reqs = decode_reqs + prefill_reqs
    layout = BatchLayout.capture(
        reqs, ("decode",) * len(decode_reqs) + ("prefill",) * len(prefill_reqs)
    )
    return Batch(
        reqs=reqs,
        phase="mixed" if decode_reqs and prefill_reqs else ("decode" if decode_reqs else "prefill"),
        layout=layout,
    )
