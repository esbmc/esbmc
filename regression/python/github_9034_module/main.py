def g():
    return h


def h():
    return 2


k = g()
assert k() == 2
