x = nondet_int()
__ESBMC_assume(x >= 1 and x <= 64)
p = 2**x
r = bool(x % p)
assert not r
