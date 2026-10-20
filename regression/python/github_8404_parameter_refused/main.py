# The parameter holds a function returning str or one returning int.
def s():
    return "a"


def i():
    return 1


def call(g):
    return g()


if nondet_bool():
    f = s
else:
    f = i
r = call(f)
assert r == "a" or r == 1
