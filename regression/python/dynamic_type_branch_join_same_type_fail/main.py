# When types do not diverge, a genuinely wrong post-join assertion is still
# detected on both paths.

cond = nondet_bool()
if cond:
    x = 1
    y = 2
else:
    x = 3
    y = 4
z = x + y
assert z == 999
