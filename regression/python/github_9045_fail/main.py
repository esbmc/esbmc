# Issue #9045: importing typing.Any must not erase a set element's type
import random
from typing import Any

n = random.randint(0, 10)
v = {n}
for i, e in enumerate(v):
    assert isinstance(e, str)
