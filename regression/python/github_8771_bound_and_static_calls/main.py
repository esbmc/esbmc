class C:

    def func(self):
        return 1

    @staticmethod
    def s():
        return 2

    @classmethod
    def k(cls):
        return 3


c = C()
assert C.func(c) == 1
assert c.func() == 1
assert C.s() == 2
assert C.k() == 3
