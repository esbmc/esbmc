# A zero or False value is not None for Optional[float] and Optional[bool] (#8016).
from typing import Optional


def f(s: int) -> Optional[float]:
    if s == 200:
        return 0.0
    return None


def b(s: int) -> Optional[bool]:
    if s == 200:
        return False
    return None


x: Optional[float] = f(nondet_int())
if x is not None:
    assert x == 0.0
y: Optional[bool] = b(nondet_int())
if y is not None:
    assert not y
z: Optional[float] = f(200)
assert z is not None
t: Optional[bool] = b(200)
assert t is not None
