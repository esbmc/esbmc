class A:

    def f(self):
        pass


def g(x):
    x.f()


g(A())
