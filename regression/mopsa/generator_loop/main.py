n = 10
rand1 = [0, n - 1]
rand2 = [-1000, 1000]

def rand_range():
  i = 0
  while i < n:
    k = rand1[0]
    yield k
    i = i + 1

l1 = [rand2[0]] * n    
g = rand_range()
l2 = []

for x in g:
  y = l1[x]
  l2.append(y)
  pass


