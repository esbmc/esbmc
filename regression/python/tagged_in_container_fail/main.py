import random


def f(flag):
    if flag:
        return 42
    return "unknown"


flag = random.random() < 0.5
c = f(flag)
assert "n" in c
