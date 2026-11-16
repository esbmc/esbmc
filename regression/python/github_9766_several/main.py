l = [[] for _ in range(4)]
l[1].append(10)
l[1].append(11)
l[2].append(20)
l[0].append(1)
assert l[1] == [10, 11]
assert l[2] == [20]
assert l[0] == [1]
assert l[3] == []
