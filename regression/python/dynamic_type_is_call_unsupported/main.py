# `is` against a call is refused rather than folded to a constant: folding
# would discard the call, and with it the exception it raises.

def f():
    raise ValueError("boom")

cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
r = x is f()
assert True
