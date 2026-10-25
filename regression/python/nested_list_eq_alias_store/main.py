a = [1]
x = [a]
y = x
y[0] = [2]
assert x == [[2]]
