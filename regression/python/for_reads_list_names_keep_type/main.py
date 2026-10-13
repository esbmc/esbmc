x = nondet_int()
s = "ab"
xs = [x, s]
ki = 0
ks = 0
for b in xs:
    if isinstance(b, int):
        ki += 1
    if isinstance(b, str):
        ks += 1
assert ki == 1
assert ks == 1
