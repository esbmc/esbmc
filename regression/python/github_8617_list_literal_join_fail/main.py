a = nondet_int()
l = [0]
if a > 0:
    l = [1]
else:
    l = [2]
assert l[0] == 2
