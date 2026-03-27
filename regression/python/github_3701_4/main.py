n = 10
rand1 = [0, n - 1]
rand2 = [-1000, 1000]


def my_range():
    i = 0
    while i < n:
        k = rand1[0]
        assert 0 <= k < n
        yield k
        i = i + 1
    assert i == n


l1 = [rand2[0]] * n

assert len(l1) == n
assert all(l1[i] == l1[0] for i in range(n))  # now works

g = my_range()
l2 = []

j = 0
while j < n:
    assert 0 <= j < n
    x = next(g)
    assert 0 <= x < n
    j = j + 1

assert j == n
assert len(l2) == 0
