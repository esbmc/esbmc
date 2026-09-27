# A branch-local rebinding keeps the parameter untyped (#7745).
x = 1
if x > 0:
    x = 2.5
g = lambda n: n + 0
assert g(x) == 2
