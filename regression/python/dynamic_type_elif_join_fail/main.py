cond1 = nondet_bool()
cond2 = nondet_bool()
if cond1:
    x = 1
elif cond2:
    x = 2
else:
    x = "a"

assert x == 1
