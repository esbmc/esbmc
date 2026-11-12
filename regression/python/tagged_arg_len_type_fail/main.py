use_int = nondet_bool()
if use_int:
    r = 12
else:
    r = "ab"
assert len(r) == 2
