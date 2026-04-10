n = 10
rand1 = [0, n - 1]
rand2 = [-1000, 1000]

def range():
  i = 0
  while i < n:
    k = rand1[0]
    yield k
    i = i + 1

l1 = [rand2[0]] * n
g = range()
l2 = []

j = 0
while j < n:
  x = next(g)
  j = j + 1
  pass
