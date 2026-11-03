a, *b = 1, 2, 2.5
assert b[0] == 2
assert b[1] == 2.5

c, *d = 0, "s", 7
assert d[0] == "s"
assert d[1] == 7
