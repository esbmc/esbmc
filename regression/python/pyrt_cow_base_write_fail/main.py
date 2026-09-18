class A:
    pass


class B(A):
    pass


def add(self, other):
    return 42


A.__add__ = add
assert B() + B() == 0
