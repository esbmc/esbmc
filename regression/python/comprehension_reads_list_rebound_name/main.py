a = 1
xs = [a, a + 1]
a = 10
ys = [b for b in xs]
assert ys == [1, 2]
