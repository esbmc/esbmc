x: int = nondet_int()
y: int = nondet_int()
z: int = nondet_int()

__ESBMC_assume(x > 0 and y > 0 and z > 0)
__ESBMC_assume(x < 10 and y < 20 and z < 30)
assert (False)
