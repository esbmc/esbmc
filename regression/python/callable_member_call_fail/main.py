from typing import Callable


def act(x: int) -> int:
    return x + 1


class Holder:
    def __init__(self, fn: Callable[[int], int]) -> None:
        self.fn: Callable[[int], int] = fn

    def run(self, v: int) -> int:
        return self.fn(v)


h: Holder = Holder(act)
assert h.run(1) == 3
