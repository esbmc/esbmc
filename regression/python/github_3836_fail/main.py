# A failing case: accessing first element of empty filtered list
def f(a, k):
    p = a[0]
    b = [x for x in a if x > p]
    n = len(a) - len(b)
    if k >= n:
        return f(b, k - n)
    return p

# f([3], 1) must fail: b is empty, k=1 >= n=1, then f([], 1) has a[0] OOB
assert f([3], 1) == 3
