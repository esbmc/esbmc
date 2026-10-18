class A:
    def __add__(self, other):
        return 1


class B(A):
    def __add__(self, other):
        return 2


def add(self, other):
    return 3


A.__add__ = add
assert A() + A() == 3
assert B() + B() == 2
