import random


def f():
    return 1


def g():
    return "a"


call = lambda h: h()
assert call(f) == 1
assert call(g) == "a"
flag = random.random() < 0.5
r = call(f if flag else g)
if flag:
    assert r == 1
else:
    assert r == "a"
