# GitHub #9133: a count is an int, so only the second assertion fails.
t = (1, 2, 2)
i = t.index(2)
assert isinstance(i, int)
c = t.count(2)
assert isinstance(c, str)
