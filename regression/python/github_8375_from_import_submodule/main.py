from pkg import mod

a = mod.A()
assert a.f() == 1
assert mod.g(2) == 3
