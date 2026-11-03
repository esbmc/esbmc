# Issue #9045: importing typing.Any must not erase a set element's type
import random
from typing import Any

n = random.randint(0, 10)
v = {n}
assert all(isinstance(e, int) for e in v)

for e in v:
    assert isinstance(e, int)

for i, e in enumerate(v):
    assert isinstance(e, int)

for e in reversed(sorted(v)):
    assert isinstance(e, int)
