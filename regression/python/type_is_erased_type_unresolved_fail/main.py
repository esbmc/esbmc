n = nondet_int()


def g():
    return n


def func(a):
    return a()


b = func(g)
if type(b) is int:
    assert False
