# through returns pick's function, which hands back its argument, so the final
# call goes through f or g.
def f():
    return "a"


def g():
    return 1


def through(h):
    return h


def pick():
    return through


c = nondet_bool()
r = pick()(f if c else g)()
assert isinstance(r, str) == c
if c:
    assert r == "a"
else:
    assert r == 1
