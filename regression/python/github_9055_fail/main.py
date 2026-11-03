# #9055
class A:
    def f(self):
        return 1


def f(x):
    if x > 0:
        res = A().f()
    else:
        res = None
    return res


a = f(0)
assert a == 1
