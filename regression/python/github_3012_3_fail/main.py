import random
from typing import Literal


def foo(s: Literal["bar"] | None) -> None:
    assert s is not None
    assert s == "bar"


x = random.randint(0, 1)

if x:
    foo("bar")
else:
    foo(None)
