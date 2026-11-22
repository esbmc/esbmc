from mod import is_one

cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"

assert is_one(x) == cond
