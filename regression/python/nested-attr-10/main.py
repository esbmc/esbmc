# Test case 10: Variable shadowing with nested attribute access
x: int = 100  # Global variable


class Inner:

    def get_val(self) -> int:
        return 42


class Outer:

    def __init__(self) -> None:
        self.inner: Inner = Inner()

    def test_shadowing(self) -> int:
        x: int = self.inner.get_val()
        return x


o = Outer()
local_res: int = o.test_shadowing()

assert local_res == 42
assert x == 100
