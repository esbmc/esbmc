from typing import Callable


def act(x: int) -> int:
    return x + 1


class Holder:
    def __init__(self, fn: Callable[[int], int]) -> None:
        self.fn = fn


h = Holder(act)
assert h.fn(1) == 2
