xs = [1, 2]
ys = list(xs)
ys.append(3)
ys[0] = 9
assert len(xs) == 2
assert xs[0] == 1
assert len(ys) == 3
assert ys[0] == 9
