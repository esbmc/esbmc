# 10390: a signed int literal returned beside None keeps the int type.
import random


def pick(x):
    if x < 0:
        return -1
    return None


x = random.randint(-20, -1)
v = pick(x)
if x < 0:
    assert v != -1
