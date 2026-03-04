from typing import Optional

d: dict[str, Optional[int]] = {}
r: Optional[int] = d.setdefault("k")

# key must have been inserted even when no default is given
assert "k" in d
