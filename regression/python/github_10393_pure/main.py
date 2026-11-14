# Lists of one numeric type keep the typed models.
a = sorted([3, 1, 2])
assert a[0] == 1 and a[1] == 2 and a[2] == 3
assert type(a[0]) is int

b = sorted([2.5, 0.5, 1.5], reverse=True)
assert b[0] == 2.5 and b[1] == 1.5 and b[2] == 0.5
