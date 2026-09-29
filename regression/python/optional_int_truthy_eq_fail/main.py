# None is not equal to 0 (#8016).
from typing import Optional

a: Optional[int] = None
assert a == 0
