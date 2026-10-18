d = {1: 2}
ks = list(d.keys())
ks.append(3)
assert len(ks) == len(d)
