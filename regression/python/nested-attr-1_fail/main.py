# Test case 1: Multi-level nested attribute access - failing case
class A:
    def get_value(self) -> int:
        return 42

class B:
    def __init__(self) -> None:
        self.a: A = A()

    def get_a(self) -> A:
        return self.a

class C:
    def __init__(self) -> None:
        self.b: B = B()

    def test(self) -> int:
        # Test self.b.a.get_value() - two levels of nesting
        # No type annotation to trigger type inference
        result = self.b.a.get_value()
        return result

c = C()
# This should fail: expected 42 but assertion is wrong
assert c.test() == 100


