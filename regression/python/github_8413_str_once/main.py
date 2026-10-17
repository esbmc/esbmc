a = nondet_int()
s = "P"
assert a <= 0 or s == "P"
assert a > 0 or s != "N"
