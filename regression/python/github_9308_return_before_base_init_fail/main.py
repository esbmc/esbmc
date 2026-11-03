# C.__init__ returns before calling B.__init__ when a is falsy, so C(0) has no
# a and reading it raises AttributeError (#9308).
class B:

    def __init__(self, a):
        self.a = a


class C(B):

    def __init__(self, a):
        if not a:
            return
        super().__init__(a)


c = C(0)
r = c.a
