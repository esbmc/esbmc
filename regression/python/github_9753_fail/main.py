c = {1: 1 + 1}
for v in c.values():
    assert isinstance(v, int)
c[1] = "x"
for v in c.values():
    assert isinstance(v, int)
