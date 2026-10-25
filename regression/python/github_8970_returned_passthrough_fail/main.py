# As github_8970_returned_passthrough, but the int path asserts what is false.
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
if not n:
    assert r != 1
