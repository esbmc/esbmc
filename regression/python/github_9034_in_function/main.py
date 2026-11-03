def g():
    return h


def h():
    return 2


def run():
    k = g()
    return k()


assert run() == 2
