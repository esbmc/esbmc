# Generalisation of #5915: `(a OP b) ** n` with a non-constant base was mis-folded
# to a compile-time constant (0 or 1) because handle_power read the empty value
# string of the non-constant base as 0. The base must stay symbolic.
a = nondet_int()
b = nondet_int()
__ESBMC_assume(a == 7 and b == 2)
assert (a + b) ** 2 == 81
assert (a - b) ** 2 == 25
assert (a * b) ** 2 == 196
assert (a // b) ** 2 == 9
assert (a % b) ** 3 == 1
