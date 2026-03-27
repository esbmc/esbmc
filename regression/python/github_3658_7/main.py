from typing import Optional

d: dict[str, Optional[int]] = {}
r: Optional[int] = d.setdefault("k")

# returned value must match the stored value
assert d["k"] == r

d2: dict[str, Optional[int]] = {}
r2: Optional[int] = d2.setdefault("k", None)

# key must have been inserted when an explicit None default is given
assert "k" in d2

# explicit-None defaults must be stored and returned consistently
assert d2["k"] == r2

