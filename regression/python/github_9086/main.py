# GitHub #9086: int() and float() of None raise TypeError.
from typing import Optional

s: Optional[str] = None
try:
    int(s)
    assert False
except TypeError:
    pass

try:
    float(s)
    assert False
except TypeError:
    pass

try:
    float(None)
    assert False
except TypeError:
    pass

try:
    int(None)
    assert False
except TypeError:
    pass

c = nondet_bool()
t: Optional[str] = None
if c:
    t = "7"
try:
    int(t)
    ok = True
except TypeError:
    ok = False
assert ok == c
