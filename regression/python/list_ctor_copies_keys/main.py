d = {1: 2}
ks = list(d.keys())
ks.append(3)
assert len(d) == 1
assert len(ks) == 2
