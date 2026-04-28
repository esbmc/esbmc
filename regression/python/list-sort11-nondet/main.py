x = nondet_int()
y = nondet_int()
z = nondet_int()

__ESBMC_assume(x > y)
__ESBMC_assume(y > z)
__ESBMC_assume(z < 1000)

l = [x, y, z]
l.sort()
assert l[0] == z
assert l[1] == y
assert l[2] == x
