# i % 3 reaches index 2 of a two-element list: IndexError, not a verdict.
def f():
    return 1


def g():
    return "a"


fs = [f, g]
i = nondet_int()
r = fs[i % 3]()
