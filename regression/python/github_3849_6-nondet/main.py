# min/max with mixed int and float nondet arguments
a: int = nondet_int()
b: float = nondet_float()
c: int = nondet_int()

__ESBMC_assume(a >= 0 and a <= 10)
__ESBMC_assume(b >= 0.0 and b <= 10.0)
__ESBMC_assume(c >= 0 and c <= 10)

m = min(a, b, c)

# Result must be <= every argument (after promotion to float)
assert m <= a
assert m <= b
assert m <= c

mx = max(a, b, c)

# Result must be >= every argument
assert mx >= a
assert mx >= b
assert mx >= c
