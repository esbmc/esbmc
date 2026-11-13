a = [1, 2]
b = [1.0, 2.0]
assert a == b
assert sorted(a) == b
c = [2, 1]
assert c != b
f = [1.0, 2.0]
assert a == f
assert not (a != f)
