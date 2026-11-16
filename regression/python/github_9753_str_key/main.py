k = "key"
c = {k: -3}
for key, v in c.items():
    assert isinstance(key, str)
    assert isinstance(v, int)
    assert v == -3
