# C.__init__ does not call A.__init__, so a C instance has no a, and the
# read through B.get raises AttributeError.
class A:

    def __init__(self, v):
        self.a = v


class B(A):

    def get(self):
        return self.a


class C(B):

    def __init__(self, v):
        self.v = v


b = C(1) if nondet_bool() else B(1)
r = b.get()
