# Issue #8017: the None path of an Optional[Tuple[...]] function is seen.
from typing import Optional, Tuple


def pair(s: int) -> Optional[Tuple[int, int]]:
    if s == 200:
        return (1, 2)
    return None


r: Optional[Tuple[int, int]] = pair(nondet_int())
assert r is not None
