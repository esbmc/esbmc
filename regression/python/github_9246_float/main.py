# A float key stored with d[x] = v is iterated as a float (#9246).
d = {}
d[1.5] = 1
d[2.5] = 3
d[1.5] = 4
n = 0
for k in d:
    assert k == 1.5 or k == 2.5
    assert d[k] == 4 or k == 2.5
    n += 1
assert n == 2
assert 1.5 in d
