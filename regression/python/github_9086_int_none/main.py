from typing import Optional

# GitHub #9086: int() of None raises TypeError; of an Optional[str] holding
# text, parses it.


def f(s: Optional[str]) -> int:
    try:
        return int(s)
    except TypeError:
        return -1


assert f(None) == -1
assert f("3") == 3

try:
    x = int(None)
except TypeError:
    x = 2
assert x == 2

b = False and int(None)
assert b == False
