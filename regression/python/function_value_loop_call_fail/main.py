def f():
    return 1


def g():
    return 2


fs = [f, g]
for h in fs:
    r = h()
    assert r == 1
