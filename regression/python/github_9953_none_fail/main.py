from typing import Optional

try:
    from esbmc import nondet_bool
except ImportError:
    def nondet_bool() -> bool:
        return False


value: Optional[tuple[int, int]] = None
if nondet_bool():
    value = (1, 2)
assert len(value) == 2
