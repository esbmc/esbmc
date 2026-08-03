# Storing callables in a class instance field, as opposed to a module-level
# variable (which callable3 and friends already cover). This is the last of
# the four features regression/python/concurrency_fail waits on (#4566) --
# threading.Thread subclassing, queue.Queue and random.choice over a filtered
# list all work now.
#
# ESBMC currently aborts rather than reporting anything: the list element type
# comes out as empty/void, so dereferencet::construct_from_array asks it for a
# width and the symbolic_type_excp escapes uncaught.
from typing import List, Callable


def cb(x: int) -> int:
    return x + 1


class Holder:
    def __init__(self) -> None:
        self.fns: List[Callable[[int], int]] = []

    def add(self, f: Callable[[int], int]) -> None:
        self.fns.append(f)


h = Holder()
h.add(cb)
assert h.fns[0](1) == 2
