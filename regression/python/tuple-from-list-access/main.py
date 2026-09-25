# Split from tuple-from-list: len, subscript, and iteration over a tuple
# built from a list.

u = tuple([4, 5, 6])
assert len(u) == 3
assert u[0] == 4
assert u[2] == 6
total = 0
for x in u:
    total = total + x
assert total == 15

print("ok")
