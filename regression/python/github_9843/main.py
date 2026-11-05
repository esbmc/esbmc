# Issue #9843: a tuple stored on one branch into an Optional[tuple] variable
# is wrapped, so reading an element after the None check is well typed.
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
    assert value[0] < 2.0

pair: Optional[tuple[str, int]] = None
if nondet_bool():
    pair = ("ab", 2)
if pair is not None:
    assert pair[1] == 2
