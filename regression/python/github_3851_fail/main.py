# Global variable accessed from method - assertion should fail
class A:
    def pre(self) -> bool:
        return counter > 0


counter: int = 0

a = A()
assert a.pre()
