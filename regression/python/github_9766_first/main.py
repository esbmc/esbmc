l = [[] for _ in range(3)]
l[0].append(3)
assert len(l[0]) == 1
assert l[0][0] == 3
assert len(l[1]) == 0
assert len(l[2]) == 0
