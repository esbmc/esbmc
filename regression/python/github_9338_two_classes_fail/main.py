# As github_9338_two_classes, but the value asserted for B is false.
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


a = f(A())
b = f(B())
assert isinstance(a, int)
assert isinstance(b, int)
assert b == 2
