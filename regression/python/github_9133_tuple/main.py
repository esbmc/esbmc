# GitHub #9133: tuple.index() and tuple.count() return an int.
t = (1, 2, 2)
i = t.index(2)
assert isinstance(i, int)
assert i == 1
c = t.count(2)
assert isinstance(c, int)
assert c == 2
assert not isinstance(c, str)
assert isinstance(t.index(1), int)
