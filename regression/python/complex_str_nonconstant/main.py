# str() on non-constant complex (result of arithmetic) returns placeholder.
z1 = complex(1, 1)
z2 = complex(2, 3)
w = z1 + z2
s = str(w)
assert s == "(complex)"
