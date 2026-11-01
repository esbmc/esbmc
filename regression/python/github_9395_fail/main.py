# int() of a float wider than 32 bits keeps its value (#9395).
a: float = nondet_float()
__ESBMC_assume(a >= 0.0 and a <= 1e10)
assert int(a) < 2147483648
