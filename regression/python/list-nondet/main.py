A = [nondet_int()]

def f():
    if A[0] == 1:
        return 0
    return None

r = f()
assert r is None or A[r] != 1 or A[r] == 1
