import random

from typing import Any


def foo(x: int) -> Any:
    if x == 4:
        return True
    elif x == 2:
        return True
    else:
        return 3


x = random.randint(0, 10)
y = foo(x)
assert y == True or y == 2
