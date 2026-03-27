def f(a, k):
    p = a[0]
    b = [x for x in a if x > p]
    n = len(a) - len(b)
    if k >= n:
        return f(b, k - n)
    return p


assert f([1, 2, 3], 1) == 2
