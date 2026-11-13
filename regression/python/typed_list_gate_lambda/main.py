l = [1, 2]
l.append(4)
first_and_last = lambda: l[0] + l[2]
assert first_and_last() == 5
assert l[1] == 2
