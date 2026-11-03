# Issue #8200: the default comes from the class attribute K, not from 3.
class C:
    K: int = 2

    def f(self, x=K):
        return x


assert C().f() == 3
