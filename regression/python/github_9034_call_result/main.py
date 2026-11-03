def g():
    return h


def h():
    return 2


assert g()() == 2
