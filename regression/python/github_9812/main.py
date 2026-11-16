import random

flag = random.random() < 0.5


def f(flag, n):
    if flag:
        return "Hello"
    return n + 1


b = f(flag, 2)
if not flag:
    assert b == 3
