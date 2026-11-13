def f():
    return "a"


def g():
    return 1


def h():
    return 2.5


def k():
    return [1, 2]


a, *b, c = f, g, h, k
d = a()
e = b[0]()
x = b[1]()
y = c()
assert d == "a"
assert e == 1
assert x - e == 1.5
assert y == [1, 2]
