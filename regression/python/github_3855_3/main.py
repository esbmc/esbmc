# float() applied to various integer expressions (BinOps)
a = 10
b = 3

# Addition
r1 = float(a + b)
assert r1 == 13.0

# Subtraction
r2 = float(a - b)
assert r2 == 7.0

# Multiplication
r3 = float(a * b)
assert r3 == 30.0

# Nested arithmetic
r4 = float(a + b * 2)
assert r4 == 16.0

# int() of float() roundtrip
r5 = int(float(a + b))
assert r5 == 13
