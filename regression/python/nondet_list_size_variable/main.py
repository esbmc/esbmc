# A variable is a size, not a generator, even when its name starts with
# `nondet_`: the old parser read `nondet_size` as an element generator, reverted
# the bound to the default 8 and appended `nondet_size()` (esbmc/esbmc#7575).
nondet_size = 5
x = nondet_list(nondet_size)
assert len(x) <= 5
