c = nondet_bool()
xs = [1, 2]
if c:
    xs = [3, 4]
assert xs[0] == 1
