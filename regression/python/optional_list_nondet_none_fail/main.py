from typing import List, Optional


def f(s: int) -> Optional[List[int]]:
    if s == 200:
        return [1]
    return None


r: Optional[List[int]] = f(nondet_int())
assert r is not None
