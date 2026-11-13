import random


def f():
    return 42


def g():
    return "a"


flag = random.random() < 0.5
if flag:
    ls = [[f], g]
else:
    ls = [[g]]
r = ls[0][0]()
if flag:
    assert r == 42
    assert ls[1]() == "a"
else:
    assert r == "a"
