# esbmc/esbmc#7575: a nondet_list returned from a function was never expanded,
# so the caller saw a list whose elements were all the same value.
def make():
    return nondet_list(3)

x = make()
if len(x) == 2:
    assert x[0] == x[1]
