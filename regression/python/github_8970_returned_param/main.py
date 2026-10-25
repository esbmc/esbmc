# The result of h() is returned from call, so call dispatches on the function
# it was given. Each path then has the type of its own result.
def f():
    return "a"


def g():
    return 1


def call(h):
    return h()


c = nondet_bool()
r = call(f if c else g)
if c:
    assert isinstance(r, str)
    assert r == "a"
else:
    assert isinstance(r, int)
    assert r + 1 == 2
assert call(f) == "a"
assert call(g) == 1
