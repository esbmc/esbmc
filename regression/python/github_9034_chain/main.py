def a():
    return b


def b():
    return c


def c():
    return 5


assert a()()() == 5
