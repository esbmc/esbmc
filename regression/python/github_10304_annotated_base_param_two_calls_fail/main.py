class A:

    def m(self, x: int) -> int:
        return x


class B(A):

    def m(self, x: int) -> int:
        return 10 // x


def run(a: A) -> int:
    return a.m(0)


run(A())
run(B())
