l = [3, 1, 2]
assert l.index(1) == 1
assert l.count(3) == 1

l.insert(0, 9)
assert l[0] == 9
assert len(l) == 4

assert l.pop() == 2
assert l.pop(0) == 9
assert len(l) == 2

l.remove(1)
assert len(l) == 1
assert l[0] == 3

m = [1, 2]
m.extend([3, 4])
assert len(m) == 4
assert m[3] == 4

c = m.copy()
c.append(5)
assert len(c) == 5
assert len(m) == 4

s = [3, 1, 2]
s.sort()
assert s[0] == 1
assert s[1] == 2
assert s[2] == 3

m.clear()
assert len(m) == 0
