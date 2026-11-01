mt = {1: "hi"}
mt2 = mt.copy()
for k in mt2:
    assert isinstance(k, int)
for k, v in mt2.items():
    assert isinstance(k, int)
    assert isinstance(v, str)
