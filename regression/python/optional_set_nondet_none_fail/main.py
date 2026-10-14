from typing import Set, Optional


def f(s: int) -> Optional[Set[int]]:
    if s == 200:
        return {1}
    return None


r: Optional[Set[int]] = f(nondet_int())
assert r is not None
