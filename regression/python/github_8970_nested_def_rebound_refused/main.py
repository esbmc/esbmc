# inner is rebound to helper after its def, so make_s returns helper, not the def.
def helper():
    return "b"


def make_s():
    def inner():
        return "a"

    inner = helper
    return inner


def make_i():
    def other():
        return 1

    return other


def pick(c):
    if c:
        return make_s
    return make_i


c = nondet_bool()
r = pick(c)()()
if c:
    assert r == "b"
