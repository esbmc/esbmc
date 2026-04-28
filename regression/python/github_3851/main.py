class A:
    def pre(self) -> bool:
        return counter > 0


counter: int = 1

a = A()
assert a.pre()
