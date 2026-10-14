# A call of a call of a call's result (#8474).
def f():
    return 3


def g():
    return f


def h():
    return g


assert h()()() == 4
