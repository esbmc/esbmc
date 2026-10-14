# A narrowed Optional[int] passed to an int parameter passes its value (#8016).
from typing import Optional


def g(s: int) -> Optional[int]:
    if s == 200:
        return 5
    return None


def h(x: int) -> int:
    return x


v: Optional[int] = g(200)
if v is not None:
    assert h(v) == 5
