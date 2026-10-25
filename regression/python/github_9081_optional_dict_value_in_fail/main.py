# A dict stored into an Optional[dict[...]] name keeps the None path (#9081).
from typing import Optional

inner: Optional[dict[str, str]] = None
if nondet_bool():
    d: dict[str, str] = {"t": "x"}
    inner = d
assert "t" in inner
