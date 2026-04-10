x = []
x.extend([1] + r for r in [[]])

assert x == [[]]
