class A:
    def f(self, x):
        return x + 1


def f(cls):
    return 100


def call(cls):
    return cls.f(1)


assert call(A()) == 2
