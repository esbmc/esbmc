def gen():
  j = 0
  while j < 10:
    j = j + 1
    yield j

b = gen()
i = 0
while i < 5:
  i = next(b)

res = i
