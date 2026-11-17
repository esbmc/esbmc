# h may hold g, for which h(1, y=2) is 21.
def f(x, y):
    return x - y


def g(x, y):
    return x + y * 10


h = f if nondet_bool() else g
r = h(1, y=2)
assert r == -1
