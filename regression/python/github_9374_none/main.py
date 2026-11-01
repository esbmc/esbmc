# bool() and truth tests of an Optional[str] holding None (#9374).
from typing import Optional


def as_bool(s: Optional[str]) -> bool:
    return bool(s)


def as_cond(s: Optional[str]) -> bool:
    if s:
        return True
    return False


def empty_or_none(b: bool) -> Optional[str]:
    if b:
        return ""
    return None


s: Optional[str] = None
if nondet_bool():
    s = ""
assert not as_bool(s)
assert not as_cond(s)
assert not bool(empty_or_none(nondet_bool()))
