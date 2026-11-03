def f(x, y=5):
    return x + y


def g(*args):
    return f(*args)


def h(*args):
    return g(*args)


assert h(1, 2) == 6
