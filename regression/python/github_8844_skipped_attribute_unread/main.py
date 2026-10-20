# C.__init__ skips A.__init__, but nothing reads a, so nothing is refused.
class A:

    def __init__(self, v):
        self.a = v


class C(A):

    def __init__(self, v):
        self.v = v


c = C(1)
assert c.v == 1
