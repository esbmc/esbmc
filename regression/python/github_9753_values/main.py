c = {1: 1 + 1}
for v in c.values():
    assert isinstance(v, int)
    assert v == 2
