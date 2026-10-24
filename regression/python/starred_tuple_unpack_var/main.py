t = (1, 2, 3, 4)
a, *b, c = t
assert a == 1 and c == 4
assert b == [2, 3]
b.append(5)
assert b == [2, 3, 5]
assert len(b) == 3

*init, last = t
assert init == [1, 2, 3]
assert last == 4
assert t == (1, 2, 3, 4)
