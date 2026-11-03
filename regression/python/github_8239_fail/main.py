# Bytes branches of different lengths have no common type: refused rather
# than truncating the longer branch, which proved len(x) == 2 (#8239).
c: bool = nondet_bool()
a = b"ab"
b = b"abcd"
x = a if c else b
assert len(x) == 2
