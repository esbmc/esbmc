# isinstance after conjugate: result is still complex.
z = complex(1, 2)
c = z.conjugate()
assert isinstance(c, complex)
assert not isinstance(c, int)
