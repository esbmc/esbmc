xs = [1, 5, 6]
while any(v > 3 for v in xs):
    xs.pop()
assert xs == [1]
