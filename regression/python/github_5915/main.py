# https://github.com/esbmc/esbmc/issues/5915
# (a // b) ** 2 was constant-folded to 0 at conversion time because handle_power
# read the empty value string of the non-constant floor-division base as 0.
a = nondet_int()
b = nondet_int()
__ESBMC_assume(a == 10 and b == 3)
result = (a // b) ** 2
assert result == 9
