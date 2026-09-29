# Optional[int] values produced by calls (the Optional<T> struct, not a
# retyped None literal) through truthiness, ==, returns and dict arguments
# (#8016).
from typing import Optional


def g(s: int) -> Optional[int]:
    if s == 1:
        return 0
    if s == 2:
        return 5
    return None


def as_int(x: Optional[int]) -> int:
    if x is None:
        return -1
    return x


def first(d: dict[str, Optional[int]]) -> int:
    n = 0
    for v in d.values():
        if v is None:
            n += 1
    return n


none = g(0)
zero = g(1)
five = g(2)
if none:
    assert False
if zero:
    assert False
if not five:
    assert False
assert none != 0
assert zero == 0
assert not (none == zero)
assert as_int(none) == -1
assert as_int(five) == 5
assert first({"a": None, "b": 3}) == 1
