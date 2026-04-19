
x = nondet_int()
y = nondet_int()
z = nondet_int()

__ESBMC_assume(x > y)
__ESBMC_assume(y > z)

t = (x, y, z)

assert min(t) == z
assert max(t) == x
