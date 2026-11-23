def same(n: int, m: int) -> bool:
    return n is m


a = int("100")
b = int("100")
assert a is b
assert not (a is int("1001"))
n = nondet_int()
m = nondet_int()
__ESBMC_assume(-5 <= n and n <= 256)
assert same(n, m) == (n == m)
k = nondet_int()
assert k is k
