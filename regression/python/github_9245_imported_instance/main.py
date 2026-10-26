from mod import smth

v = nondet_int()
a = smth.func(v)
assert a == v
assert type(a) == int
assert type(a) is int
assert isinstance(a, int)
assert a + 1 == v + 1
