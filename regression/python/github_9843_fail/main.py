# Issue #9843: the stored tuple's element is read, not a default.
from typing import Optional

try:
    from esbmc import nondet_bool
except ImportError:
    def nondet_bool() -> bool:
        return True


value: Optional[tuple[float, float]] = None
if nondet_bool():
    value = (1.5, 2.5)
if value is not None:
    assert value[0] > 2.0
