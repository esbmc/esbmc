a = [True, False]
b = [False]
a.extend(b)
assert len(a) == 3
assert a[2] is False
expected = [True, False, False]
assert a == expected
c = [True]
c.extend([False, True])
assert c != a
other = [True, False, True]
assert c == other
a.extend(a)
assert len(a) == 6
assert a[3] is True
