# Both functions return an int, so call has nothing to dispatch.
def inc(x):
    return x + 1


def dbl(x):
    return x * 2


def call(h, x):
    return h(x)


c = nondet_bool()
n = nondet_int()
__ESBMC_assume(n > 0 and n < 100)
r = call(inc if c else dbl, n)
assert r == (n + 1 if c else n * 2)
