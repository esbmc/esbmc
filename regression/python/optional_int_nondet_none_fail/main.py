# Optional[int] takes the Optional<T> struct `int | None` does, so a zero value
# and None stay apart and `is None` is decidable (#8016).
from typing import Optional


def g(s: int) -> Optional[int]:
    if s == 200:
        return 5
    return None


v: Optional[int] = g(nondet_int())
assert v is not None
