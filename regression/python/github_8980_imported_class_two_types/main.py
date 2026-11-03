import mod

a = mod.A(nondet_int())
assert isinstance(a.b, int)
s = mod.A("s")
assert isinstance(s.b, str)
