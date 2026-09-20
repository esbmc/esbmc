def is_one(v):
    return v == 1

cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"

assert is_one(x) == cond
assert is_one(1)
assert not is_one("a")
