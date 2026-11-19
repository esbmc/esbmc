import random


def func(x):
    return x


a = func(1)
b = func("y")
n = 3
s = "a"
c = func(n) if random.random() < 0.5 else func(s)
assert c == n or c == s
