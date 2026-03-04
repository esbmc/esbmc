A = [1]

def f():
    if A[0] == 1:
        return 0
    return None

r = f()
if r is not None:
    assert A[r] == 1
