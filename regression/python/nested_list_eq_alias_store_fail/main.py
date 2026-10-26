a = [1]
x = [a]
y = x
y[0] = [(1, 2)]
assert x == [[1]]
