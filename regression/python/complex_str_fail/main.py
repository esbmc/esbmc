# str(z) with wrong expected value should fail verification.
z = complex(1, 2)
s = str(z)
assert s == "(1+3j)"
