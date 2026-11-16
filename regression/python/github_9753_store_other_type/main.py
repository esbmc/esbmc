# A value of another type stored into a dict literal is read back through
# .values() with its own type, for a literal and a computed value (#9753).
c = {1: 2}
c[1] = "x"
for v in c.values():
    assert v == "x"

d = {1: 1 + 1}
d[1] = "y"
for v in d.values():
    assert v == "y"
