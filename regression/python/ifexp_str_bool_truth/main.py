from typing import Optional


def f(c: bool, s: Optional[str]) -> int:
    if (s if c else False):
        return 1
    return 0


s = ""
c = True
n = 0
if (s if c else True):
    n = 1
assert n == 0

while (s if c else True) and n < 2:
    n += 1
assert n == 0

assert f(True, "a") == 1
assert f(True, "") == 0
assert f(True, None) == 0
assert f(False, "a") == 0


class A:
    def __init__(self) -> None:
        self.s: Optional[str] = None


a = A()
if (a.s if c else True):
    n = 1
assert n == 0
if a.s:
    n = 1
assert n == 0
