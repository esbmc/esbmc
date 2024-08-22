x = int(0)
assert x.bit_length() == 0

y = int(16)
assert y.bit_length() == 5

z = int(255)
assert z.bit_length() == 8

x = 5
assert (x-1).bit_length() == 3


def foo(a: int) -> int:
  return int((a - 1).bit_length())

