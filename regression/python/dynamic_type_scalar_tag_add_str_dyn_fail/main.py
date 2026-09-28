cond1 = nondet_bool()
cond2 = nondet_bool()
if cond1:
    x = 1
    y = 2
elif cond2:
    x = "a"
    y = "bb"
else:
    x = "ccc"
    y = "d"
z = x + y
assert z == "wrong"
