# isinstance(z, complex) works after arithmetic operations.
z1 = complex(1, 2)
z2 = complex(3, 4)
w = z1 + z2
assert isinstance(w, complex)
assert not isinstance(w, int)
assert not isinstance(w, float)
