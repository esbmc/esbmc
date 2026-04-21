z = complex(1, 2)
z = 3.14
assert z == complex(3.14, 2.0)  # wrong: imag should be 0.0 after promotion
