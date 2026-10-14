from typing import Optional

x: int = nondet_int()
r: Optional[str] = "OK" if x == 200 else None
assert r is not None or x != 200
