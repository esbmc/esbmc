l = [5, 10, 15, 20]
x = l.pop(-2)  # Remove second-to-last element
assert x == 15
assert len(l) == 3
assert l[0] == 5
assert l[1] == 10
assert l[2] == 20
