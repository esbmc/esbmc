c = nondet_bool()
xs = [1, 2]
if c:
    xs = [3, 4]
for i in range(2):
    xs = [5, 6]
assert xs[0] == 5
