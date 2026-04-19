l = [10, 20, 30, 40]
x = l.pop(1)  # Remove element at index 1
assert x == 20
assert len(l) == 3
assert l[0] == 10
assert l[1] == 30
assert l[2] == 40
