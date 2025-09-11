import random


def foo(x: int, y: int) -> int:
    if x:
        if y:
            return 1
        else:
            return None
    return 2


x = random.randint(0, 1)
y = random.randint(0, 1)
foo(x, y)
