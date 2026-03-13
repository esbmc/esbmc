# Non-lambda ternary with None branch - None on the else side
from typing import Optional

def get_val(n: int) -> Optional[int]:
    return n * 2 if n > 0 else None

r1 = get_val(5)
assert r1 is not None

r2 = get_val(-3)
assert r2 is None
