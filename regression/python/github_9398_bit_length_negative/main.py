# Issue #9398: bit_length of a negative int is the bit length of abs(n).
x: int = nondet_int()
__ESBMC_assume(x == -37)
assert x.bit_length() == 6

y: int = nondet_int()
__ESBMC_assume(y < 0)
assert y.bit_length() >= 1

m: int = nondet_int()
__ESBMC_assume(m == -9223372036854775807 - 1)
assert m.bit_length() == 64

z: int = nondet_int()
__ESBMC_assume(z >= -256 and z <= -1)
r: int = z.bit_length()
assert (1 << (r - 1)) <= -z
assert -z < (1 << r)
