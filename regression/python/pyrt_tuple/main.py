t = (1, 2, 3)
assert len(t) == 3
assert t[0] == 1
assert t[2] == 3
a, b, c = t
assert a == 1
assert c == 3
assert t == (1, 2, 3)
assert t != (1, 2)
assert isinstance(t, tuple)
assert type(t) is tuple
s = 0
for v in t:
    s = s + v
assert s == 6
