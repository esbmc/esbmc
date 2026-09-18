class A:
    pass


class B(A):
    def __sub__(self, other):
        return 1


def add(self, other):
    return 2


A.__add__ = add
assert B() + B() == 1
