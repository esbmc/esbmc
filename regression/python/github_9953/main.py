# Issue #9953: len, for and unpacking read the tuple held by an
# Optional[tuple] variable, not the Optional wrapper.
from typing import Optional

try:
    from esbmc import nondet_bool
except ImportError:
    def nondet_bool() -> bool:
        return True


value: Optional[tuple[int, int]] = None
if nondet_bool():
    value = (1, 2)
if value is not None:
    assert len(value) == 2
    n = 0
    for v in value:
        n += 1
    assert n == 2
    a, b = value
    assert a == 1 and b == 2
    c, *rest = value
    assert c == 1
    t: tuple[int, int] = value
    assert len(t) == 2

pair: Optional[tuple[float, float]] = None
if nondet_bool():
    pair = (1.5, 2.5)
if pair is not None:
    x, y = pair
    assert x + y == 4.0
    assert len(pair) == 2
