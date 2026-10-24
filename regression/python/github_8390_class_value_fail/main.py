u = int
if nondet_bool():
    u = str
assert not (u is str)
