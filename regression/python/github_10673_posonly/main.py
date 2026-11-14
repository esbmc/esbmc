# a is positional-only in f, so h(a=1, b=2) raises TypeError when h is f.
def g(a, b):
    return a + b


def f(a, /, b):
    return a + b


h = g if nondet_bool() else f
assert h(a=1, b=2) == 3
