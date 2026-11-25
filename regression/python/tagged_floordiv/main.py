import random


def f(flag):
    if flag:
        return 7
    return "seven"


flag = random.random() < 0.5
x = f(flag)
if flag:
    assert 14 // x == 2
    assert -15 // x == -3
    assert 15 // -x == -3
    assert x // 2 == 3
    assert -x // 2 == -4
    assert x // x == 1
y = 7 if flag else "seven"
if flag:
    assert y // 2 == 3
    assert y // x == 1
