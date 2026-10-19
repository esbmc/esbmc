from esbmc import nondet_int, __ESBMC_assume

n = nondet_int()
__ESBMC_assume(0 <= n and n < 62)
assert 2**n == (1 << n)

k = nondet_int()
__ESBMC_assume(-3 <= k and k <= -1)
v = 2**k
assert v == 0.5 or v == 0.25 or v == 0.125
assert v * (2**(-k)) == 1.0
