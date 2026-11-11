# bool() of an Optional[str] holding None (#9374).
from typing import Optional


def as_bool(s: Optional[str]) -> bool:
    return bool(s)


s: Optional[str] = None
if nondet_bool():
    s = "a"
x = as_bool(s)
assert not x
