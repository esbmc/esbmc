from pkg import LIMIT, mod as m, util

a = m.A()
assert a.f() + util.g(LIMIT) == 5
