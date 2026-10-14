cond1 = nondet_bool()
cond2 = nondet_bool()
if cond1:
    if cond2:
        x = 1
    else:
        x = 2
else:
    x = "a"

assert x == 1
