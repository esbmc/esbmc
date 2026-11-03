# C.__init__ skips A.__init__ but sets a itself.
class A:

    def __init__(self, v):
        self.a = v


class C(A):

    def __init__(self, v):
        self.a = v + 1


c = C(1)
assert c.a == 2
