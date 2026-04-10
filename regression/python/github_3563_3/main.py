A = [nondet_int()]

def f():
    if A[0] == 1:
        return 42
    return

r = f()
if r is not None:
    assert r == 42
