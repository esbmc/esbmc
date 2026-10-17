# A None returned where `-> int` is declared is reported, not read as 0 (#8016).
from typing import Optional


def g(s: int) -> Optional[int]:
    if s == 1:
        return 0
    return None


def raw(x: Optional[int]) -> int:
    return x


r = raw(g(0))
assert r == 0
