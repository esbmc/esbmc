# Bytes branches of one length still select between their values (#8239).
c: bool = nondet_bool()
a = b"ab"
b = b"cd"
x = a if c else b
assert len(x) == 2
assert x[0] == 97 or x[0] == 99
