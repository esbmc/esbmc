# 10390: a signed int literal returned beside None keeps the int type.
import random


def pick(x):
    if x < 0:
        return -1
    return None


def half(x):
    if x < 0:
        return -1.5
    return None


def mixed(x):
    if x < 0:
        return -2
    return 2.5


x = random.randint(-20, 20)
v = pick(x)
w = mixed(x)
h = half(x)
if x < 0:
    assert v == -1
    assert w == -2
    assert h == -1.5
else:
    assert v is None
    assert w == 2.5
    assert h is None
