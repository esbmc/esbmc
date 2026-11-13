import random


def f(flag):
    if flag:
        return 7
    return "seven"


flag = random.random() < 0.5
x = f(flag)
y = 3 if flag else "three"
if flag:
    assert x * 2 == 14
    assert 3 * x == 21
    assert x * y == 21
    assert -x * y == -21
    assert x % 3 == 1
    assert -x % 3 == 2
    assert x % -3 == -2
    assert 15 % x == 1
    assert -15 % x == 6
    assert x % y == 1
