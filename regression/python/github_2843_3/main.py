import random
from typing import Any


def foo(x: int) -> Any:
    if x > 0:
        return 3.14
    else:
        return 5


x = random.randint(-10, 10)

assert foo(x) == 3.14 or foo(x) == 5
