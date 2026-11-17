class A:
    def int(self):
        return 1

    def list(self):
        return 2

    def set(self):
        self.v = 3

    def str(self):
        return self.int() + int("4")

    def type(self):
        return 6

    def object(self):
        return 7

    def divmod(self):
        return 8


class B(A):
    pass


a = A()
assert a.int() == 1
assert a.list() == 2
a.set()
assert a.v == 4
assert a.str() == 5
assert a.type() == 6
assert a.object() == 7
assert a.divmod() == 8
assert B().list() == 2
assert len(list((1, 2))) == 2
assert divmod(7, 2) == (3, 1)
