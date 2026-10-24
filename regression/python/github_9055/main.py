# #9055: f returns an int from a method call or None.
class A:
    def f(self):
        return 1


def f(x):
    if x > 0:
        res = A().f()
    else:
        res = None
    return res


a = f(1)
assert a == 1
assert f(0) is None
