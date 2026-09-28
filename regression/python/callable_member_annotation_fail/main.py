from typing import Callable


def act(x: int) -> int:
    return x + 1


class Holder:
    def __init__(self, f: Callable[[int], int], tag: int) -> None:
        self.f: Callable[[int], int] = f
        self.tag: int = tag


h: Holder = Holder(act, 7)
assert h.tag == 8
