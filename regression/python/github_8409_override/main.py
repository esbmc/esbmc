# The declaration the receiver's class resolves to decides the binding: an
# ordinary method overriding a @staticmethod takes self, and a @staticmethod
# overriding an ordinary method does not (#8409).
class B:
    @staticmethod
    def f(x: int) -> int:
        return x

    def g(self, x: int) -> int:
        return x + 100


class D(B):
    def f(self, x: int) -> int:
        return x + 1

    @staticmethod
    def g(x: int) -> int:
        return x + 2


assert B().f(1) == 1
assert D().f(1) == 2
assert B().g(1) == 101
assert D().g(1) == 3
