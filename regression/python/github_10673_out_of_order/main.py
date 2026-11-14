# h(y=2, x=1) binds x to 1 and y to 2 whichever function h holds.
def f(x, y):
    return x - y


def g(x, y):
    return x + y * 10


h = f if nondet_bool() else g
r = h(y=2, x=1)
assert r == -1 or r == 21
