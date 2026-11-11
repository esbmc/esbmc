# A negative int literal returned beside None stays an int (esbmc/esbmc#10390).
import random


def pick(x: int):
    if x < 0:
        return -1
    return None


def big(x: int):
    if x < 0:
        return -9007199254740993
    return None


x = random.randint(-20, 20)
v = pick(x)
w = big(x)
if x < 0:
    assert v == -1
    assert w == -9007199254740993
else:
    assert v is None and w is None
