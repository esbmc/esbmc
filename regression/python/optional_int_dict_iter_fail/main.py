# values() of dict[str, Optional[int]] yields None for "a" (#8016).
from typing import Optional

d: dict[str, Optional[int]] = {"a": None, "b": 5}
for v in d.values():
    assert v is not None
