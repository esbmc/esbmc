n = nondet_int()


def g():
    return n


def func(a):
    return a()


b = func(g)
# A lambda argument is not followed, so the type of `a()` stays lost.
func(lambda: n)
assert isinstance(b, int)
