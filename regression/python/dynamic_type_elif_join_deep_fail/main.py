cond1 = nondet_bool()
cond2 = nondet_bool()
cond3 = nondet_bool()
if cond1:
    x = 1
elif cond2:
    x = 2
elif cond3:
    x = "a"
else:
    x = "b"

assert x == 1
