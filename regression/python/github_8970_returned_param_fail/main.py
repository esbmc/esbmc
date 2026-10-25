# As github_8970_returned_param, but the int path asserts what is false.
def f():
    return "a"


def g():
    return 1


def call(h):
    return h()


c = nondet_bool()
r = call(f if c else g)
if not c:
    assert r != 1
