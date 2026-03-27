# Symbolic exponent fixed at -10 via nondet + assume,
# verifying pow() is used (not the constant 1 fallback)
n: int = nondet_int()
__ESBMC_assume(n == -10)
result = 2**n
assert result < 0.001
