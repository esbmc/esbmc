# Optional[int] is encoded as an int pointer holding the value, so a zero value
# and None coincide and `is None` cannot be decided from it.
from typing import Optional


def g(s: int) -> Optional[int]:
    if s == 200:
        return 5
    return None


v: Optional[int] = g(nondet_int())
if v is not None:
    assert v + 1 == 6
w: Optional[int] = g(200)
assert w is not None and w == 5
