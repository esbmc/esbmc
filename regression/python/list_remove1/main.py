l = [1, 2, 3, 2, 4]
l.remove(2)
assert l[0] == 1
assert l[1] == 3  # first '2' removed; second '2' still there
assert l[2] == 2
assert l[3] == 4
