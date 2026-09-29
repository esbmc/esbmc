# A function returning T | None, Union[T, None] or an Optional parameter unannotated
# can return None (#8016).
from typing import Optional, Union


def p(v: Optional[int]) -> int | None:
    return v


def u(v: Optional[int]) -> Union[int, None]:
    return v


def n(v: Optional[int]):
    return v


a = p(None)
b = u(None)
c = n(None)
assert a is None
assert b is None
assert c is None
x = p(3)
y = u(0)
z = n(4)
assert x == 3
assert y == 0
assert z == 4
