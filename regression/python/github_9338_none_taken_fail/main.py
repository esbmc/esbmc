# As github_9338_none_taken, but the None path is asserted to be 1.
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


class C:
    pass


assert f(C()) == 1
