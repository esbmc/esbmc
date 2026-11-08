def g():
    return 1


def h():
    return 2


def func(n):
    if n == 0:
        return None
    if n == 1:
        return g
    return h


assert func(2)() == 1
