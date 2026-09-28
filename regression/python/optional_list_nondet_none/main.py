from typing import List, Optional


def f(s: int) -> Optional[List[int]]:
    if s == 200:
        return [1]
    return None


r: Optional[List[int]] = f(nondet_int())
if r is not None:
    assert r[0] == 1
    assert len(r) == 1
