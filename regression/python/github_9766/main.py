l = [[] for _ in range(3)]
l[0].append(3)
assert len(l[0]) == 1
assert l[0][0] == 3
assert len(l[2]) == 0

m = [[] for x in range(2)]
m[0].append(2.5)
m[1].append(1.5)
m[1].extend([4.0])
assert m[0][0] == 2.5
assert m[1] == [1.5, 4.0]
m[1].pop()
assert len(m[1]) == 1

k = [[1, 2] for _ in range(2)]
k[0].remove(1)
assert k[0] == [2]
assert k[1] == [1, 2]
