import random


def add(x, y):
    return x + y


use_int = random.random() < 0.5
if use_int:
    x, y = 1, "b"
else:
    x, y = "a", 2
r = add(x, y)
