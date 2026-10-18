class A:
    pass


class B(A):
    pass


def add(self, other):
    return 7


patched = nondet_bool()
if patched:
    A.__add__ = add
assert B() + B() == 7
