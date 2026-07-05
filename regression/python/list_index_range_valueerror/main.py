# list.index(x, start[, end]) now raises a catchable ValueError when x is not
# present in l[start:end], instead of an uncatchable property violation.
l = [1, 2, 3, 4]
try:
    l.index(9, 1, 3)
    assert False
except ValueError:
    pass

# Present in the whole list but outside the [start:end] window -> ValueError.
m = [1, 2, 1, 2]
try:
    m.index(2, 0, 1)
    assert False
except ValueError:
    pass

# A present element in the window still returns its index (in the original list).
assert l.index(3, 1, 4) == 2
assert m.index(2, 2) == 3
assert l.index(3, -3, -1) == 2
