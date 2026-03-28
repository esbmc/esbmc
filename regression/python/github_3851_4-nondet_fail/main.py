# Nondet int global with no assumption — method can return False,
# so the assertion is reachable and must fail.

class A:
    def pre(self) -> bool:
        return counter > 0


counter: int = nondet_int()

a = A()
assert a.pre()
