# C.__init__ calls the base __init__, so a is set.
class A:

    def __init__(self, v):
        self.a = v


class C(A):

    def __init__(self, v):
        super().__init__(v)
        self.v = v


c = C(1)
assert c.a == 1
