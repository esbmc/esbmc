class A:

    def m(self, x: int) -> int:
        return x


class B(A):

    def m(self, x: int) -> int:
        return 10 // x


def run(a: A) -> int:
    assert isinstance(a, B)
    return 1


run(A())
