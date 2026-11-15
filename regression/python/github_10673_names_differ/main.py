# g has no parameter x, so h(x=1) raises TypeError when h is g.
def f(x):
    return x


def g(y):
    return y


h = f if nondet_bool() else g
assert h(x=1) == 1
