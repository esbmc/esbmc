from esbmc import nondet_float, __ESBMC_assume

x = nondet_float()
__ESBMC_assume(x >= 0.0)
__ESBMC_assume(x <= 1.0)
assert x <= 2.0
