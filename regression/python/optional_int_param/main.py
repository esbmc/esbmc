# An Optional[int] argument reaches an Optional[int] parameter intact (#8016).
from typing import Optional


def g(s: int) -> Optional[int]:
    if s == 200:
        return 5
    return None

def h(x: Optional[int]) -> int:
    if x is None:
        return -1
    return x


v: Optional[int] = g(nondet_int())
r = h(v)
assert r == 5 or r == -1
w: Optional[int] = g(200)
assert h(w) == 5
assert h(5) == 5 and h(0) == 0 and h(None) == -1
