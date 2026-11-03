first, *rest = 1, 2, 3
assert first == 1
assert rest == [2, 3]

a, *b, c = 1, 2, 3, 4
assert b == [2, 3]
assert a == 1 and c == 4

x, *empty, y = 5, 6
assert x == 5 and y == 6
assert empty == []
assert len(empty) == 0

*init, last = (7, )
assert init == []
assert last == 7
