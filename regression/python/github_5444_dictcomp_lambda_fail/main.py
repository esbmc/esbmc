d = {1: 0, 2: 0}
r = {k: (lambda k: k)(10) for k in d}
assert r[1] == 1
