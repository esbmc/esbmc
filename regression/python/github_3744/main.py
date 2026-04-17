class C:
    def f(self) -> int:
        return 42

c = C()
g = c.f
result = g()
assert result == 42
