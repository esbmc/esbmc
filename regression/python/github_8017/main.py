# Issue #8017: `is None` is decidable on Optional[Tuple[...]] and
# Optional[Dict[...]] values.
from typing import Dict, Optional, Tuple


def pair(s: int) -> Optional[Tuple[int, int]]:
    if s == 200:
        return (1, 2)
    return None


def table(s: int) -> Optional[Dict[str, int]]:
    if s == 200:
        return {"a": 1}
    return None


assert pair(200) is not None
assert pair(1) is None
assert table(200) is not None
assert table(1) is None
p: Optional[Tuple[int, int]] = pair(200)
if p is not None:
    assert p[0] + p[1] == 3
