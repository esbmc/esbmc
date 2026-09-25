from typing import Any


class C:
    def __init__(self, v: int):
        self.v = v

    def get(self) -> int:
        return self.v


x: Any = C(7)
assert x.v == 7
assert x.get() == 7

f: Any = lambda a: a + 1
assert f(2) == 3
