s: set[int] = { 1, 2, 1 }
assert 1 in s

a = 0
for x in s:
  a = a + x

assert a == 4
