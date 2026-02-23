l = [1, 2, 3]
a = l.pop()
assert a == 3
b = l.pop()
assert b == 2
c = l.pop()
assert c == 1
assert len(l) == 0
