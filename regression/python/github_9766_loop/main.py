l = []
for _ in range(3):
    l.append([])
l[0].append(1)
assert len(l[0]) == 1
assert len(l[2]) == 0
