# isinstance(z, complex) should fail when assertion is wrong.
z = complex(1, 2)
assert not isinstance(z, complex)
