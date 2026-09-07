# Companion to nondet_list_elem_type: a float list can hold 0.5, so the same
# assertion must be falsified. Proving it would mean float elements had
# silently degraded to ints.
x: list[float] = nondet_list(3, nondet_float())
__ESBMC_assume(len(x) >= 1)
assert x[0] != 0.5
