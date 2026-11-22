import random


def f():
    return 42


def g():
    return "a"


flag = random.random() < 0.5
ls = [[f if flag else g], g]
r = ls[0][0]()
if flag:
    assert r == 43
else:
    assert r == "a"
