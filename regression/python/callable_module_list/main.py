# The module-level counterpart of callable_class_field: calling a callable held
# in a list element. This shape reports `got invalid code for function
# ...$list_elem$` and exits rather than aborting, so it pins the reachable half
# of #6640 -- if the abort in callable_class_field is ever turned into a
# diagnostic, both shapes should end up here.
from typing import List, Callable


def cb(x: int) -> int:
    return x + 1


fns: List[Callable[[int], int]] = []
fns.append(cb)

assert fns[0](1) == 2
