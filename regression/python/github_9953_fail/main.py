from typing import Optional

try:
    from esbmc import nondet_bool
except ImportError:
    def nondet_bool() -> bool:
        return True


value: Optional[tuple[int, int]] = None
if nondet_bool():
    value = (1, 2)
if value is not None:
    a, b = value
    assert len(value) == 2
    assert b == 1
