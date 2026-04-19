n = 10
rand = [0, n - 1]

def rand_range():
  i = 0
  while i < n:
    k = rand[0]
    yield k
    i += 1

l1 = [0] * n
l2 = []

gen = rand_range()
# i = 0
# while i < n:
#   x = next(gen)
#   assert 0 <= x
#   assert x < len(l1)
#   y = l1[x] # safe index access
#   l2.append(y)
#   i += 1

for x in range(n - 2):
    x = next(gen)
    assert 0 <= x
    assert x < len(l1)
    y = l1[x] # safe index access
    l2.append(y)

  
