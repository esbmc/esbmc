# The module-level counterpart of callable_class_field: calling a callable held
# in a list element (#6640, #8505).
from typing import List, Callable


def cb(x: int) -> int:
    return x + 1


fns: List[Callable[[int], int]] = []
fns.append(cb)

assert fns[0](1) == 2
