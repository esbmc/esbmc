a = 4
c = {1: a}
vs = list(c.values())
v = vs[0]
assert isinstance(v, int)
assert v == 4
