"""Optional pipeline trace for the demo dashboard.

TraceRecorder is threaded through the query pipeline as trace=None by
default — every existing caller (/chat, eval.runner) is unaffected. Only
/chat/trace passes one in, so the recording overhead only exists when
someone is actually asking to see it.
"""

import time
from dataclasses import dataclass, field


@dataclass
class TraceStep:
    stage: str
    elapsed_ms: float
    data: dict


@dataclass
class TraceRecorder:
    steps: list[TraceStep] = field(default_factory=list)
    _start: float = field(default_factory=time.perf_counter)

    def record(self, stage: str, **data) -> None:
        elapsed = (time.perf_counter() - self._start) * 1000
        self.steps.append(TraceStep(stage=stage, elapsed_ms=round(elapsed, 2), data=data))
