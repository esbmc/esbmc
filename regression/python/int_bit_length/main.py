a = int(0)
assert a.bit_length() == 0

b = int(16)
assert b.bit_length() == 5

c = int(255)
assert c.bit_length() == 8

d = 5
assert (d-1).bit_length() == 3


def foo(x: int) -> int:
  return int((x - 1).bit_length())


e = int(5)
f = e.bit_length()
assert f == 3

g = e.bit_length() - 1
assert g == 2

def foo(x:int) -> None:
  y = x.bit_length()