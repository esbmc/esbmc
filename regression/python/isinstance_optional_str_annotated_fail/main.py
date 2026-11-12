# 10434 control: an annotated Optional[str] that holds None is not a str.
import random
from typing import Optional

s: Optional[str] = "a" if random.random() < 0.0 else None
assert isinstance(s, str)
