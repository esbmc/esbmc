l = [1, 2, 3, 4, 5]
l.pop()
l.pop(0)
l.append(99)
assert len(l) == 4
assert l[0] == 2
assert l[1] == 3
assert l[2] == 4
assert l[3] == 99
