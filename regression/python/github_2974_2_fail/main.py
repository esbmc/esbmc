class A:

    def __init__(self) -> None:
        pass

    def method_a(self) -> int:
        return 100


class B:

    def __init__(self) -> None:
        self.a: A = A()


class C:

    def __init__(self) -> None:
        self.b: B = B()


class D:

    def __init__(self) -> None:
        self.c: C = C()

    def test(self) -> int:
        return self.c.b.a.method_a()  # Three-level attribute chain


# Test three-level attribute chain - should fail assertion
d = D()
result = d.test()
assert result == 99  # Wrong value - should fail
