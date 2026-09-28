cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"

d = {'a': lambda v: v == 1}
assert d['a'](x)
