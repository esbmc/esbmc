# Test: super().method() with return type annotation (the exact issue from #3838)
class A:

    def f(self) -> str:
        return "A"


class B(A):

    def f(self) -> str:
        x = super().f()
        return x


b = B()
result = b.f()
assert result == "A"
