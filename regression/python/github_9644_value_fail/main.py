mt = {1: "hi"}
mt2 = mt.copy()
for k, v in mt2.items():
    assert isinstance(v, int)
