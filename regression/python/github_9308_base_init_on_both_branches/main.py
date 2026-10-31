# Both branches of C.__init__ call B.__init__, so every C has a (#9308).
class B:

    def __init__(self, a: int):
        self.a = a


class C(B):

    def __init__(self, a: int):
        if a:
            super().__init__(a)
        else:
            B.__init__(self, 5)

    def cfunc(self) -> int:
        return self.a


x = nondet_int()
c = C(x)
assert c.cfunc() != 0
