def f():
    return 1


def g():
    return 2


ls = [f, g]
b = ls[1]
assert b is f
