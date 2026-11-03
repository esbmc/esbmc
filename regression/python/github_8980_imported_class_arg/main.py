import mod
from mod import P

for i in range(2):
    a = mod.A(nondet_int())
    if not isinstance(a.b, int):
        raise TypeError("lost")
    p = P(nondet_float())
    if not isinstance(p.x, float) or not isinstance(p.y, float):
        raise TypeError("lost")
