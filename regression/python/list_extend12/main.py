def f(k):
    if k == 0:
        return [[]]

    ret = []
    for r in f(k - 1):
        ret.extend([1] + r)
    return ret


f(1)
