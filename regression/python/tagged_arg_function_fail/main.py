import random


def add(x, y):
    return x + y


use_int = random.random() < 0.5
if use_int:
    x, y = 1, 2
else:
    x, y = "a", "b"
r = add(x, y)
if use_int:
    assert r == 4
else:
    assert r == "ba"
