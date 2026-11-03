class A:
    def who(self) -> int:
        return 1


class B(A):
    pass


class C(A):
    def who(self) -> int:
        return 2


class D(B, C):
    def call(self) -> int:
        return self.who()


assert D().call() == 1
