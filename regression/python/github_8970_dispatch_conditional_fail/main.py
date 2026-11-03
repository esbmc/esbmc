# As github_8970_dispatch_conditional, but the int path asserts what is false.
def f():
    return "a"


def g():
    return 1


c = nondet_bool()
h = f if c else g
r = h()
if not c:
    assert r != 1
