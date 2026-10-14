import threading


# Regression test for a soundness defect in the subclass lowering:
# the construction-rewrite predicate previously fired on any call
# carrying a ``target=`` keyword, even when the call was a user
# subclass whose own ``__init__`` happened to take a ``target``
# parameter. The rewrite then stripped every keyword, silently
# dropping the user's data. With the predicate tightened to
# ``_is_thread_constructor``, the subclass constructor passes through
# untouched and the kwarg reaches ``__init__`` as written.
class Worker(threading.Thread):
    def __init__(self, target: int = 0) -> None:
        super().__init__()
        # Stash the kwarg on the instance so we can observe it.
        self.target_val: int = target

    def run(self) -> None:
        # Spawned-thread body is a no-op: this test exercises the
        # constructor path. The lowering must spawn ``run`` via the
        # trampoline without re-touching the construction site's
        # kwargs.
        pass


w: Worker = Worker(target=42)
w.start()
w.join()

# Reading ``w.target_val`` after the join confirms the kwarg reached
# ``__init__`` rather than being silently dropped.
assert w.target_val == 42
