# str() of complex constructed with only one argument (imaginary defaults to 0).
z = complex(5)
s = str(z)
assert s == "(5+0j)"
