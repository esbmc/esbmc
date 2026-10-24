def a():
    return 1


def b(x=a):
    return x


def g():
    return h


def h():
    return 2


f = b()
assert f() == 1
k = g()
assert k() == 2
