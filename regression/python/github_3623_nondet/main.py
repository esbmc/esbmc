lst = [nondet_int(), nondet_int(), nondet_int()]

start = nondet_int()
end = nondet_int()

__ESBMC_assume(0 <= start <= end <= len(lst))

sub = lst[start:end]

assert len(sub) == end - start
