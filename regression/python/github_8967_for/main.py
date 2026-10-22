xs = [1, 2]
ys = [3, 4]
z = zip(xs, ys)
s = 0
for a, b in z:
    s += a * b
assert s == 11
