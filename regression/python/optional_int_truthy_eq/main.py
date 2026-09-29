# An optional is truthy only when it holds a truthy value, and None equals only
# None (#8016).
from typing import Optional

a: Optional[int] = None
b: Optional[int] = 0
c: Optional[int] = 5
assert not a and not b and c
assert a != 0 and b == 0 and not (a == b)
f: Optional[bool] = True
if not f:
    assert False
d: dict[str, Optional[int]] = {"k": None}
d["k"] = 0
assert d["k"] is not None
assert d["k"] == 0
