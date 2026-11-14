# Issue #10222: a symbolic exponent may be negative while the base is zero.
n: int = nondet_int()
b: float = nondet_float()
__ESBMC_assume(b == 0.0)
x = b ** n
