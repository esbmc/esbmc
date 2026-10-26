# A float key stored with d[x] = v is iterated as a float (#9246).
d = {}
d[1.5] = 1
for k in d:
    assert k == 2.5
