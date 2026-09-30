# Optional[bool] returns None for every s but 200 (#8016).
from typing import Optional


def b(s: int) -> Optional[bool]:
    if s == 200:
        return False
    return None


y: Optional[bool] = b(nondet_int())
assert y is not None
