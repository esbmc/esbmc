import random


def f(flag):
    if flag:
        return 42
    return "unknown"


flag = random.random() < 0.5
c = f(flag)
if not flag:
    assert "kno" in c
    assert c in "unknown!"
    assert "z" not in c
