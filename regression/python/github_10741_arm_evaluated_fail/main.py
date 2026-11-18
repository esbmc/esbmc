# k() raises ValueError when flag holds.
def call(a):
    return a()


def k():
    raise ValueError("k")


def g():
    return 7


r = call(k() if nondet_bool() else g)
assert r == 7
