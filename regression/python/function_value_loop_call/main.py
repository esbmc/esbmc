def f():
    return 1


def g():
    return 2


fs = [f, g]
total = 0
for h in fs:
    r = h()
    total += r
assert total == 3
