c = nondet_bool()
raw = "25"
if c:
    raw = "15"
x = int(raw)
assert x == (15 if c else 25)
