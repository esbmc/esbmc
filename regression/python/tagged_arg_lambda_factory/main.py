import random


def make(k):
    return lambda x: x + k


use_int = random.random() < 0.5
if use_int:
    k, x = 1, 2
else:
    k, x = "a", "b"
r = make(k)(x)
if use_int:
    assert r == 3
else:
    assert r == "ba"
