# An Optional operand of str % is refused rather than unwrapped, which
# raised a TypeError CPython does not raise.
from typing import Optional


def u(b: bool) -> Optional[int]:
    return 1 if b else None


x = u(False)
assert "%s" % x == "None"
