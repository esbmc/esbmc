nested = [[nondet_int()] for _ in range(2)]

inner_idx = nondet_int()
__ESBMC_assume(0 <= inner_idx < len(nested))

if len(nested[inner_idx]) > 0:
    old_len = len(nested[inner_idx])
    val = nested[inner_idx].pop()
    assert len(nested[inner_idx]) == old_len - 1
