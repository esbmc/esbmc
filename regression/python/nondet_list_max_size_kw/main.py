# The documented `max_size=` keyword was ignored, silently capping the list at
# the default 8 (esbmc/esbmc#7575).
x = nondet_list(max_size=10)
assert len(x) <= 10
