# The stored str is not an int (#9753).
c = {1: 2}
c[1] = "x"
for v in c.values():
    assert isinstance(v, int)
