d = {0: 1}
other = {1: 2}
d.update(other)
assert d[0] == 1
assert d[1] == 2
