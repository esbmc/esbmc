from mod import is_one as one

cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"

assert one(x) == cond
