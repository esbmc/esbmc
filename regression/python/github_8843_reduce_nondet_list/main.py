from functools import reduce


def mul(x, y):
    return x * y


xs = nondet_list(nondet_int)
__ESBMC_assume(len(xs) > 0)
p = reduce(mul, xs)
assert isinstance(p, int)
