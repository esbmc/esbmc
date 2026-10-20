t = int if nondet_bool() else str
assert t is int or t is str
assert isinstance(1, t) == (t is int)
u = int
if nondet_bool():
    u = str
assert u is int or u is str
assert isinstance("a", u) == (u is str)
assert u is not None
