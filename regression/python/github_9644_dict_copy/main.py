# GitHub #9644: the copy of a dict keeps its key and value types.
mt = {1: "hi", 2: "yo"}
mt2 = mt.copy()
for k in mt2:
    assert isinstance(k, int)
    assert isinstance(mt2[k], str)
for k in mt2.keys():
    assert isinstance(k, int)
mt2[3] = "x"
assert len(mt) == 2
assert len(mt2) == 3
