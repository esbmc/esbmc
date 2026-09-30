# k(200) returns 5 through an Optional[int] local (#8016).
from typing import Optional


def g(s: int) -> Optional[int]:
    if s == 200:
        return 5
    return None

def k(s: int) -> Optional[int]:
    v: Optional[int] = g(s)
    return v


w = k(200)
assert w != 5
