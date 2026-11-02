class A:
    def f(self, x):
        return x + 3


def f(cls, x):
    return 2


def call(cls):
    return cls.f(0)


assert call(A()) == 2
