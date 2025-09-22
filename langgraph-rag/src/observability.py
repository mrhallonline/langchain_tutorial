import time, contextlib
from dataclasses import dataclass, field

@dataclass
class RunLog:
    steps: list = field(default_factory=list)
    start: float = field(default_factory=time.time)

    @contextlib.contextmanager
    def step(self, name:str, payload=None):
        t0 = time.time()
        try:
            yield
        finally:
            self.steps.append({"name":name, "ms": int((time.time()-t0)*1000), "payload": payload})

    def summary(self):
        total = int((time.time()-self.start)*1000)
        return {"total_ms": total, "steps": self.steps}