n = 10
rand = [0, n - 1]

assert n == 10
assert 0 <= rand[0] < n

def rand_range():
  i = 0
  assert i == 0
  while i < n:
    assert 0 <= i <= n
    k = rand[0]
    assert 0 <= k < n
    yield k
    i += 1
  assert i == n

l1 = [0] * n

# structural properties of l1
assert len(l1) == n
assert all(l1[i] == 0 for i in range(n))

l2 = []
assert len(l2) == 0

gen = rand_range()

count = 0

for _ in range(n - 2):

    # loop invariant
    assert 0 <= count <= n - 2
    assert len(l2) == count

    x = next(gen)

    # safety properties
    assert 0 <= x
    assert x < len(l1)

    y = l1[x]  # safe index access
    assert y == 0

    l2.append(y)
    count += 1

    # relational invariant
    assert len(l2) == count

# postconditions
assert count == n - 2
assert len(l2) == n - 2
assert all(l2[i] == 0 for i in range(len(l2)))
