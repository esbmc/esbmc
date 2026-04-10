x = nondet_int()
A = [x]

def f():
    if A[0] == 1:
        return 0
    return 2

r = f()
assert A[r] == x
