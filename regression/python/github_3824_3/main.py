# Test self.b.c.f() pattern (nested attribute on self)
class C:

    def get_value(self) -> int:
        return 7


class B:

    def __init__(self):
        self.c = C()


class A:

    def __init__(self):
        self.b = B()

    def run(self) -> int:
        return self.b.c.get_value()


a = A()
assert a.run() == 7
