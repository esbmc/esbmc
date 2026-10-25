import mod

a = mod.A(nondet_int())
assert isinstance(a.b, str)
