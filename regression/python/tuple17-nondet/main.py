x = nondet_int()
__ESBMC_assume(x > 10)
y = nondet_int()
__ESBMC_assume(y > 5)
z = nondet_int()
__ESBMC_assume(z > 3)

t = (x, y, z)

assert y in t
assert 2 not in t
