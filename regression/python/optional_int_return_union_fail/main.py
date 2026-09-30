# p(None) returns None (#8016).
from typing import Optional


def p(v: Optional[int]) -> int | None:
    return v


assert p(None) is not None
