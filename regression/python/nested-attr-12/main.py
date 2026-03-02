# Test case 12: Module-level nested attribute access (KNOWN LIMITATION)


class Inner:

    def get_val(self) -> int:
        return 100


class Outer:

    def __init__(self) -> None:
        self.inner: Inner = Inner()


o = Outer()

result: int = o.inner.get_val()

assert result == 100
