from typing import Optional

try:
    from esbmc import nondet_bool
except ImportError:
    def nondet_bool() -> bool:
        return True


value: Optional[tuple[bool, float]] = None
if nondet_bool():
    value = (True, 2.5)
if value is not None:
    assert value[1] == 3.5
