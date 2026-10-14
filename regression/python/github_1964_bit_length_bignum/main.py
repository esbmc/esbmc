x = int(0)
assert x.bit_length() == 0

y = int(16)
assert y.bit_length() == 5

z = int(2 ** 64)
assert z.bit_length() == 65
