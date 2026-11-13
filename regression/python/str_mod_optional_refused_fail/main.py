# The formatting of an Optional operand is not modelled, so the made-up
# result must not decide the assertion on it.
from typing import Optional


def u(b: bool) -> Optional[int]:
    return 1 if b else None


x = u(False)
assert "%s" % x == ""
