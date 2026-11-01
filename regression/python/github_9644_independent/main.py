mt = {1: "hi"}
mt2 = mt.copy()
mt2[2] = "there"
assert len(mt) == 1
assert len(mt2) == 2
