# #9055: the caller of f takes its Optional[int] result.
def f(x):
    if x > 0:
        res = 1
    else:
        res = None
    return res


a = f(1)
assert a == 1
