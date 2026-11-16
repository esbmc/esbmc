def label(v):
    return "seen"

def describe(v):
    return label(v)

cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
assert describe(x) == "seen"
