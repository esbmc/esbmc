# A tagged variable only ever holds a number or a string, so `is None` is
# False on every path. Issue #7075: this used to abort the frontend.

cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
assert x is not None
