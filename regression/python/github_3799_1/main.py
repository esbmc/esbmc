# Reproduces issue #3799: ESBMC crashed when a function parameter received
# objects of different subclasses due to type conflict fallback to 'int'.
# The fix infers the common base class (A) as the parameter type.


class A:

    def f(self):
        return 0


class B(A):

    def f(self):
        return 1


class C(A):

    def f(self):
        return 2


def g(x):
    y = x.f()


b = B()
c = C()

g(b)
g(c)
