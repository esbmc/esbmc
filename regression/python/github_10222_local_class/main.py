# Issue #10222: a function-local class body with no statement that can raise
# is converted.
def f() -> int:

    class A:
        """A class whose body has no statement that can raise."""
        k = 3
        pass

        def m(self) -> int:
            return 2

    return A().m()


assert f() == 2
