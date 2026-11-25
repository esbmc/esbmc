def label(v):
    w = v
    return w

def describe(v):
    return label(v)

cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
assert describe(x) == x
