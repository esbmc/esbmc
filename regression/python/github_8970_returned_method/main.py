# As github_8970_returned_param, with a str method on the tagged result. Tracked in #9148.
def f():
    return "a"


def g():
    return 1


def call(h):
    return h()


c = nondet_bool()
r = call(f if c else g)
if c:
    assert r.upper() == "A"
