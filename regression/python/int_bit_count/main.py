a = int(0)
assert a.bit_count() == 0

b = int(255)
assert b.bit_count() == 8

c = int(1024)
assert c.bit_count() == 1

# bit_count() operates on the absolute value: abs(-3) == 0b11 has two ones.
d = int(-3)
assert d.bit_count() == 2

e = 4
assert (e - 1).bit_count() == 2  # (4 - 1) == 0b11 has two ones

f = int(13)
g = f.bit_count()
assert g == 3
