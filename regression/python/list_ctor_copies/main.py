xs = [1, 2]
s = list(xs)
s.append(3)
xs[0] = 7
assert len(xs) == 2
assert len(s) == 3
assert s[0] == 1
