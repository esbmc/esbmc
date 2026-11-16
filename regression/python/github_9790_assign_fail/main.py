c = {1: 2, 2: "x"}
for k in [1, 2]:
    v = c[k]
    assert isinstance(v, int)
