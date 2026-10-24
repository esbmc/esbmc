from lib import f


def g(*args):
    return f(*args)


def total(first, *rest):
    s = first
    for r in rest:
        s += r
    return s


def h():
    return f(1, 2)
