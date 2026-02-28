A = [nondet_int()]


def g():
    return 0


def f():
    if A[0] == 1:
        return g()
    return None


r = f()
if r is not None:
    assert A[r] == 1
