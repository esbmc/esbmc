c = nondet_bool()
r = "a" if c else 1.5
assert type(r) is float
