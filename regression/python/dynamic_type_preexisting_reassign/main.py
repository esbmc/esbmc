# x already has a native type before the if/else it's reassigned inside.

x = 0
cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
assert x == 1 or x == "a"
