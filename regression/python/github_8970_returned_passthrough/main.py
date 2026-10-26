# outer passes its parameter on to call, which returns the call of it.
def f():
    return "a"


def g():
    return 1


def call(h):
    return h()


def outer(h):
    return call(h)


n = nondet_bool()
r = outer(f if n else g)
if n:
    assert isinstance(r, str)
    assert r == "a"
else:
    assert isinstance(r, int)
    assert r + 1 == 2
