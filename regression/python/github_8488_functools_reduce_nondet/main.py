from functools import reduce


def sub(x: int, y: int) -> int:
    return x - y


n = nondet_int()
__ESBMC_assume(-100 < n < 100)
assert reduce(sub, [10, n, 1]) == 9 - n
assert reduce(sub, [10, n], 1) == -9 - n
