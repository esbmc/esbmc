n = nondet_int()


def g():
    return n


def func(a):
    return a()


def run():
    # A lambda argument is followed only at module level, so here the type of `a()` stays lost.
    func(lambda: n)


b = func(g)
run()
assert isinstance(b, int)
