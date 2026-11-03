# C.__init__ calls B.__init__ only when a is truthy, so C(0) has no a and
# cfunc raises AttributeError (#9308).
class B:

    def __init__(self, a):
        self.a = a


class C(B):

    def __init__(self, a):
        if a:
            super().__init__(a)

    def cfunc(self):
        return self.a


c = C(0)
d = c.cfunc()
