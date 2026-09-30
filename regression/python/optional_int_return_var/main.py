# An Optional[int] local returned as Optional[int], and a narrowed one passed as
# int (#8016).
from typing import Optional


def g(s: int) -> Optional[int]:
    if s == 200:
        return 5
    return None

def k(s: int) -> Optional[int]:
    v: Optional[int] = g(s)
    return v


def ident(x: int) -> int:
    return x


w = k(200)
assert w is not None and w == 5
assert k(1) is None
v: Optional[int] = g(200)
if v is not None:
    assert ident(v) == 5
