class B:
    def __init__(self):
        self.b = 10


class C:
    def setc(self) -> None:
        self.c = 31


class D(B, C):
    pass


d = D()
d.setc()
assert d.b == 10
