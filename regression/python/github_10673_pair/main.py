# Both functions h may hold take (x, y), so h(1, y=2) passes 2 as y.
def f(x, y):
    return x - y


def g(x, y):
    return x + y * 10


h = f if nondet_bool() else g
r = h(1, y=2)
assert r == -1 or r == 21
