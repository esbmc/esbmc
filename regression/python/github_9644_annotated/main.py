mt: dict[int, str] = {1: "hi"}
mt2 = mt.copy()
for k in mt2:
    assert isinstance(k, int)
