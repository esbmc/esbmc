# A stored 0 is not None (#8016).
from typing import Optional

d: dict[str, Optional[int]] = {"a": 0}
v: Optional[int] = d.setdefault("a", 5)
assert v is None
