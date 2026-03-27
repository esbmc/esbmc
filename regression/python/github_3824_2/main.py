# Test with explicit type annotations on intermediate attributes
class C:

    def get_value(self) -> int:
        return 42


class B:

    def __init__(self):
        self.c: C = C()


class A:

    def __init__(self):
        self.b: B = B()


a = A()
result = a.b.c.get_value()
assert result == 42
