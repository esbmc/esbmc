x = int(0)
assert x.bit_length() == 0

y = int(16)
assert y.bit_length() == 5

z = int(2 ** 31)
assert z.bit_length() == 32
