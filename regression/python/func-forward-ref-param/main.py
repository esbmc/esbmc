def a(x):
    return b(x) * 2


def b(y):
    return y + 1


assert a(3) == 8
