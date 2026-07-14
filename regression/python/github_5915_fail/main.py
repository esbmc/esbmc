# Negative guard for #5915: result is 9, so `result == 0` must be violated.
# If the bogus compile-time fold to 0 ever returns, this would wrongly SUCCEED.
a = nondet_int()
b = nondet_int()
__ESBMC_assume(a == 10 and b == 3)
result = (a // b) ** 2
assert result == 0
