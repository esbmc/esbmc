from mod import f2
from pkg.top import g1

a = f2(nondet_int())
if not isinstance(a, int):
    raise TypeError("lost")
b = g1(nondet_int())
if not isinstance(b, int):
    raise TypeError("lost")
