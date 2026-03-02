nested = [[nondet_int(), nondet_int()] for _ in range(2)]

outer_idx = nondet_int()
__ESBMC_assume(0 <= outer_idx < len(nested))
inner = nested[outer_idx]

inner_idx = nondet_int()
__ESBMC_assume(0 <= inner_idx < len(inner))

val = inner[inner_idx]
assert val == inner[inner_idx]
