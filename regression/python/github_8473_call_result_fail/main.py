def f():
    return 1


def g():
    return 2


def pick():
    return f


assert pick() is g
