n = 10
rand1 = [0, n - 1]
rand2 = [-1000, 1000]

assert n == 10

def rand_range():
  i = 0
  while i < n:
    k = rand1[0]
    assert 0 <= k < n          # k always valid index
    yield k
    i = i + 1
  assert i == n                # loop runs exactly n times

l1 = [rand2[0]] * n

# All elements in l1 are equal and in range
assert len(l1) == n
assert all(l1[i] == l1[0] for i in range(n))
assert -1000 <= l1[0] <= 1000

g = rand_range()
l2 = []

for x in g:
  assert 0 <= x < n            # safe indexing
  y = l1[x]
  assert y == l1[0]            # all elements identical
  assert -1000 <= y <= 1000    # value stays in range
  l2.append(y)

# Postconditions
assert len(l2) == n
assert all(l2[i] == l1[0] for i in range(n))
assert l2 == l1
