# isinstance() evaluates a side-effecting Optional operand once (#9044)
import random

cnt = 0


def g() -> int:
    global cnt
    cnt += 1
    return 3


c = random.randint(0, 1)
r = isinstance(g() if c == 1 else None, (str, int))
assert cnt == c
assert r == (c == 1)
