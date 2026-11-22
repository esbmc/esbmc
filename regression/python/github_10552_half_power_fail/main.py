from esbmc import nondet_int, __ESBMC_assume

n = nondet_int()
__ESBMC_assume(1 <= n and n <= 1000)
r = int(n**0.5)
assert r * r != n
