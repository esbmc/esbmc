# The last branch stores None, the earlier ones store method results.
class A:
    def f(self):
        return 1


class B:
    def g(self):
        return 1


def f(x):
    if isinstance(x, A):
        res = x.f()
    elif isinstance(x, B):
        res = x.g()
    else:
        res = None
    return res


r = f(B())
assert isinstance(r, int)
assert r == 1
