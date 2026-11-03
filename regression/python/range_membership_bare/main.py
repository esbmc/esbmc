# `x in range(a, b)` reads the backing list's elements, unlike len(), so a
# bare range cannot be lowered to a size without contents. Non-regression
# guard for the review of #8195: giving the bare path a size and no contents
# turned this into an invalid-pointer dereference.
def f(a, b):
    assert 0 not in range(a, b)


f(1, 10)
