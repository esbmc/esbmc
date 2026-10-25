# #9055
def f(x):
    if x > 0:
        res = 1
    else:
        res = None
    return res


a = f(1)
assert a == 2
