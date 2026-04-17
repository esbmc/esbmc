nested = [[nondet_int(), nondet_int()] for _ in range(2)]

inner_idx = nondet_int()
__ESBMC_assume(0 <= inner_idx < len(nested))
inner = nested[inner_idx]

start = nondet_int()
end = nondet_int()
__ESBMC_assume(0 <= start <= end <= len(inner))

sub = inner[start:end]
assert len(sub) == end - start
