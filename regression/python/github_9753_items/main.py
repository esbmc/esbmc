a = 1
c = {a: 2.5 * 2}
for k, v in c.items():
    assert isinstance(k, int)
    assert isinstance(v, float)
    assert v == 5.0
