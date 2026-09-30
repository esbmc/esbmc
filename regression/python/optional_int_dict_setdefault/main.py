# dict[str, Optional[int]] values are optionals: 0 is not None (#8016).
from typing import Optional

d: dict[str, Optional[int]] = {"a": 7, "z": 0}
v: Optional[int] = d.setdefault("a", 5)
assert v is not None and v == 7
u: Optional[int] = d.setdefault("b", 5)
assert u is not None and u == 5
z: Optional[int] = d.setdefault("z", 5)
assert z is not None and z == 0
