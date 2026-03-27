import random


def absolute(x: int) -> int:
    if (x < 0): return -x
    else: x


x = random.randint(1, 1000)
if (x * x + x * x * x * x) == 90:
    a = absolute(absolute(x))
