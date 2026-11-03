class A:
    pass


class B(A):
    pass


def same(h) -> bool:
    return h is A


assert same(A)
assert same(B)
