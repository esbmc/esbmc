nested = [[nondet_int()] for _ in range(2)]

inner_idx = nondet_int()
__ESBMC_assume(0 <= inner_idx < len(nested))

v = nondet_int()
nested[inner_idx].append(v)

assert nested[inner_idx][-1] == v
