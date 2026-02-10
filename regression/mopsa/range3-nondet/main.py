n = nondet_int()
__ESBMC_assume(n > 0)

r = range(n)

assert len(r) > 0

x = nondet_int()
y = nondet_int()
__ESBMC_assume(x > 0)
__ESBMC_assume (x < y)

z = range(n)

assert len(z) > 0
