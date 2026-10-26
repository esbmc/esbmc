# dict[str, Optional[int]] values()/items() yield optionals: None stays None (#8016).
from typing import Optional

d: dict[str, Optional[int]] = {"a": None, "b": 5, "c": 0}
nones = 0
total = 0
for v in d.values():
    if v is None:
        nones += 1
    else:
        total += v
assert nones == 1 and total == 5
for k, w in d.items():
    if k == "a":
        assert w is None
    else:
        assert w is not None
