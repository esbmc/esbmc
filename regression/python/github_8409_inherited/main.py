# A @staticmethod inherited from a base binds no receiver through an instance of
# the subclass either (#8409).
class B:
    @staticmethod
    def ident(x: int) -> int:
        return x


class D(B):
    def call(self, y: int) -> int:
        return self.ident(y)


d = D()
assert D().ident(4) == 4
assert d.ident(6) == 6
assert d.call(7) == 7
assert D().call(8) == 8
