# C.__init__ either calls B.__init__ or raises, so every C has a (#9308).
class B:

    def __init__(self, a: int):
        self.a = a


class C(B):

    def __init__(self, a: int):
        if a >= 0:
            super().__init__(a)
        else:
            raise ValueError("negative")


c = C(4)
assert c.a == 4
