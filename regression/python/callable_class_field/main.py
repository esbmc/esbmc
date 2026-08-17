# Calling a callable that was stored in a list. Storing one works -- drop the
# call below and this verifies -- so the gap is the indirect call through the
# element, not the container (#6640).
#
# The two shapes fail differently, and only this one is a crash: through an
# instance field ESBMC aborts on to_code_type's `type.id() == typet::t_code`,
# while through a module-level list it reports `got invalid code for function
# ...$list_elem$` and exits (callable_module_list pins that one). callable3
# covers neither: it stores a callable in a module-level *variable* and then
# calls a different function directly.
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
