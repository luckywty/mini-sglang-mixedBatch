from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from minisgl.core import Req


@dataclass(eq=False)
class RequestState:
    """One slot ownership generation, shared across its prefill chunk objects.

    ForwardInput keeps these objects alive after retirement/slot reuse so an old
    in-flight result cannot emit a token, free the new owner, or reinsert a prefix.
    GPU lifetime is separately protected by the scheduler's stream dependency.
    """

    req: Req
    terminal: bool = False


@dataclass
class MixedLifecycle:
    active: Dict[int, RequestState] = field(default_factory=dict)

    def register(self, reqs: List[Req]) -> Tuple[RequestState, ...]:
        states = []
        for req in reqs:
            state = self.active.get(req.table_idx)
            if state is None:
                state = self.active[req.table_idx] = RequestState(req)
            elif state.req.uid != req.uid:
                raise RuntimeError("Request slot reused before its previous owner retired")
            state.req = req  # continuation chunks replace Req, but keep slot ownership
            states.append(state)
        return tuple(states)

    def find(self, uid: int) -> RequestState | None:
        return next((state for state in self.active.values() if state.req.uid == uid), None)

    def retire(self, state: RequestState) -> bool:
        if state.terminal:
            return False
        state.terminal = True
        assert self.active.get(state.req.table_idx) is state
        del self.active[state.req.table_idx]
        return True
