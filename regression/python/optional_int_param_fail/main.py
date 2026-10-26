# h(v) reads the argument's value, not its None flag (#8016).
from typing import Optional


def g(s: int) -> Optional[int]:
    if s == 200:
        return 5
    return None

def h(x: Optional[int]) -> int:
    if x is None:
        return -1
    return x


v: Optional[int] = g(200)
assert h(v) != 5
