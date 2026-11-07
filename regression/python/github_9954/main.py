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
    assert value[0] and value[1] == 2.5
    assert value == (True, 2.5)
    assert 2.5 in value

swapped: Optional[tuple[float, bool]] = None
if nondet_bool():
    swapped = (2.5, True)
if swapped is not None:
    assert swapped[0] == 2.5 and swapped[1]

flag: Optional[tuple[int, bool]] = None
if nondet_bool():
    flag = (3, False)
if flag is not None:
    assert flag[0] == 3 and not flag[1]

direct: Optional[tuple[bool, float]] = (False, 1.5)
assert direct[1] == 1.5
