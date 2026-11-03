# self in D is a C only when it is an E, so the call cannot be decided from
# D alone and conversion refuses it.
class A:

    def f(self):
        return 1


class C(A):
    pass


class D(A):

    def f(self):
        return super(C, self).f()


class E(D, C):
    pass


assert E().f() == 1
