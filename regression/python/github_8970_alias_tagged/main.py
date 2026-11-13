# call is invoked through the alias k, so its result is a str or an int, by
# the path.
def s():
    return "a"


def i():
    return 1


def call(h):
    return h()


k = call
c = nondet_bool()
r = k(s) if c else k(i)
assert isinstance(r, str) == c
assert r != 2
