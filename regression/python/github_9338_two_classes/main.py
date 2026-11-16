# f is called with both classes, as in the Typpete isinstance task.
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
assert a == 1 and b == 1
