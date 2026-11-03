# super(B, self) inside a subclass of B starts the lookup after B, so it
# reaches A.f rather than B.f.
class A:

    def f(self):
        return 1


class B(A):

    def f(self):
        return 2


class E(B):

    def f(self):
        return super(B, self).f()


assert E().f() == 1
