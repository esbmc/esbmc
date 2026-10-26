# Both functions return an int, so run has nothing to dispatch.
def inc(x):
    return x + 1


def dbl(x):
    return x * 2


def run(c, x):
    h = inc if c else dbl
    return h(x)


c = nondet_bool()
n = nondet_int()
__ESBMC_assume(n > 0 and n < 100)
assert run(c, n) == (n + 1 if c else n * 2)
