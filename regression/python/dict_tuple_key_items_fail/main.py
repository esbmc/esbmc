g = {('A', 'B'): 3, ('C', 'D'): 7}
n = 0
for k, w in g.items():
    n += 1
assert n == 3
