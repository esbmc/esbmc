a = int(5)
assert a.conjugate() == 5

b = int(-7)
assert b.conjugate() == -7

c = int(0)
assert c.conjugate() == 0

# Expression receiver via the BinOp dispatch path.
d = 10
assert (d - 3).conjugate() == 7
