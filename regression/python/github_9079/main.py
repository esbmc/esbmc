# GitHub #9079: membership on an Optional[str] holding None raises TypeError.
from typing import Optional


class C:
    def __init__(self) -> None:
        self.name: str = "xaby"
        self.opt: Optional[str] = None


def match(stderr: Optional[str]) -> bool:
    return 'ab' in stderr


def plain(s: str) -> bool:
    return 'ab' in s


e = ""
assert not ("a" in e)
assert match("xxaby")
assert not match("xy")
assert plain("ab")
assert None in [None, "a"]
c = C()
assert "ab" in c.name
try:
    "ab" in c.opt
    assert False
except TypeError:
    pass
try:
    match(None)
    assert False
except TypeError:
    pass
s: Optional[str] = None
try:
    "ab" in s
    assert False
except TypeError:
    pass
try:
    "ab" not in s
    assert False
except TypeError:
    pass
