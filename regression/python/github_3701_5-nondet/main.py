# symbolic size
n = nondet_int()
__ESBMC_assume(n >= 0)
__ESBMC_assume(n <= 20)

# symbolic index
rand1 = [nondet_int()]
__ESBMC_assume(0 <= rand1[0] < n if n > 0 else True)

# symbolic value
rand2 = [nondet_int()]
__ESBMC_assume(-1000 <= rand2[0] <= 1000)


def my_range():
    i = 0
    while i < n:
        k = rand1[0]
        assert 0 <= k < n
        yield k
        i = i + 1
    assert i == n


l1 = [rand2[0]] * n

# structural property
if n > 0:
    assert all(l1[i] == l1[0] for i in range(n))

g = my_range()
l2 = []

j = 0

# 🔒 critical invariant before loop
assert len(l2) == j

while j < n:

    # loop invariants
    assert 0 <= j <= n
    assert len(l2) == j

    x = next(g)
    assert 0 <= x < n

    if nondet_bool():
        y = l1[x]
    else:
        y = l1[rand1[0]]

    assert -1000 <= y <= 1000

    l2.append(y)
    j = j + 1

    # 🔒 maintain relational invariant
    assert len(l2) == j

# postconditions
assert j == n
assert len(l2) == n

if n > 0:
    assert all(l2[i] == l1[0] for i in range(n))
