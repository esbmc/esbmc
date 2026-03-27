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

    def test(self) -> int:
        return self.b.a.method_a()  # Multi-level attribute chain


# Test multi-level attribute chain
c = C()
result = c.test()
assert result == 100
