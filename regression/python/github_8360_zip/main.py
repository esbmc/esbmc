xs = [1, 3, 5]
ys = [2, 4, 6]
t = 0
for i, (a, b) in enumerate(zip(xs, ys)):
    if i == 1:
        continue
    t += i + a + b
assert t == 16
